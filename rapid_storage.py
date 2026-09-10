"""Shared RAPID records. PostgreSQL for hosting; explicit SQLite for local use.

No credentials, login state, or hypothetical scenarios are persisted. Writes use
transactions and revision checks so stale browser sessions cannot overwrite data.
"""
import copy
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd
from sqlalchemy import Column, Integer, MetaData, String, Table, Text, create_engine, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from planner_core import validate_order_ids


class StorageConflict(Exception):
    """Saved data changed since the caller loaded it."""


class StorageConfigurationError(Exception):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def frame_payload(frame):
    data = json.loads(frame.to_json(orient="split", date_format="iso"))
    return {"columns": data["columns"], "data": data["data"]}


def payload_frame(payload):
    frame = pd.DataFrame(payload["data"], columns=payload["columns"])
    for col in ("Due Date", "Expected Material Ready Date"):
        if col in frame:
            frame[col] = pd.to_datetime(frame[col], errors="coerce")
    return frame


def configuration_payload(state):
    return {"settings": dict(state["settings"]), "machine_master": frame_payload(state["machine_master"])}


def operations_payload(state):
    return {
        "planning_date": str(state["planning_date"]),
        "workers_present": int(state["workers_present"]),
        "orders": frame_payload(state["orders"]),
        "machine_today": frame_payload(state["machine_today"]),
        "production_notes": copy.deepcopy(state.get("production_notes", [])),
        "actions": copy.deepcopy(state.get("actions", [])),
    }


def next_day_payload(saved, requested_date):
    """Roll forward once from saved inputs; never add actual output twice."""
    requested = date.fromisoformat(str(requested_date))
    previous = date.fromisoformat(saved["planning_date"])
    if requested < previous:
        raise ValueError("Open a saved daily record to view an earlier date.")
    out = copy.deepcopy(saved)
    if requested == previous:
        return out
    orders = payload_frame(out["orders"])
    orders["Completed Before Today"] = orders["Completed Before Today"] + orders["Actual Production Today"]
    orders["Actual Production Today"] = 0
    orders["Workers Used Today"] = 0
    out["orders"] = frame_payload(orders)
    out["planning_date"] = str(requested)
    # Attendance and machine conditions are carried as editable assumptions.
    return out


@dataclass(frozen=True)
class Record:
    payload: dict
    revision: int
    updated_at: str
    updated_by: str


class RapidStore:
    def __init__(self, database_url):
        if not database_url:
            raise StorageConfigurationError("RAPID_DATABASE_URL is required.")
        url = make_url(database_url)
        if url.drivername in {"postgres", "postgresql"}:
            url = url.set(drivername="postgresql+psycopg")
        if url.drivername not in {"sqlite", "postgresql+psycopg"}:
            raise StorageConfigurationError("Use PostgreSQL or an explicit local SQLite database.")
        self.is_local = url.drivername == "sqlite"
        if self.is_local and url.database and url.database != ":memory:":
            Path(url.database).parent.mkdir(parents=True, exist_ok=True)
        args = {"timeout": 20} if self.is_local else {"connect_timeout": 10}
        pool_options = {"poolclass": NullPool} if self.is_local and url.database != ":memory:" else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=args, hide_parameters=True, **pool_options)
        self.metadata = MetaData()
        self.documents = Table(
            "rapid_documents", self.metadata,
            Column("name", String(40), primary_key=True),
            Column("schema_version", Integer, nullable=False),
            Column("revision", Integer, nullable=False),
            Column("payload", Text, nullable=False),
            Column("updated_at", String(40), nullable=False),
            Column("updated_by", String(200), nullable=False),
        )
        self.days = Table(
            "rapid_daily_records", self.metadata,
            Column("planning_date", String(10), primary_key=True),
            Column("payload", Text, nullable=False),
            Column("updated_at", String(40), nullable=False),
            Column("updated_by", String(200), nullable=False),
        )
        self.replans = Table(
            "rapid_replans", self.metadata,
            Column("save_id", String(36), primary_key=True),
            Column("revision", Integer, nullable=False, unique=True),
            Column("planning_date", String(10), nullable=False),
            Column("payload", Text, nullable=False),
            Column("saved_at", String(40), nullable=False),
            Column("saved_by", String(200), nullable=False),
        )
        self.config_audit = Table(
            "rapid_configuration_history", self.metadata,
            Column("revision", Integer, primary_key=True),
            Column("payload", Text, nullable=False),
            Column("saved_at", String(40), nullable=False),
            Column("saved_by", String(200), nullable=False),
        )
        self.metadata.create_all(self.engine)
        self.insert = sqlite_insert if self.is_local else pg_insert

    def close(self):
        self.engine.dispose()

    def initialize(self, configuration, operations):
        """Insert initial references only if missing; never overwrite saved data."""
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            for name, payload in (("configuration", configuration), ("operations", operations)):
                conn.execute(self.insert(self.documents).values(
                    name=name, schema_version=1, revision=0, payload=dumps(payload),
                    updated_at=now, updated_by="initial setup",
                ).on_conflict_do_nothing(index_elements=["name"]))
            self._read(conn, "configuration")
            self._read(conn, "operations")

    def _read(self, conn, name):
        row = conn.execute(select(self.documents).where(self.documents.c.name == name)).mappings().one()
        if row["schema_version"] != 1:
            raise StorageConfigurationError("Unsupported database schema. No data was changed.")
        return Record(json.loads(row["payload"]), row["revision"], row["updated_at"], row["updated_by"])

    def load(self):
        with self.engine.connect() as conn:
            return self._read(conn, "configuration"), self._read(conn, "operations")

    def _replace(self, conn, name, payload, revision, actor, now):
        result = conn.execute(update(self.documents).where(
            self.documents.c.name == name, self.documents.c.revision == revision,
        ).values(payload=dumps(payload), revision=revision + 1, updated_at=now, updated_by=actor))
        if result.rowcount != 1:
            raise StorageConflict("Another user saved changes. Reload saved data before saving again.")

    def save_configuration(self, payload, revision, actor, role):
        if role != "Admin":
            raise PermissionError("Only Admin can save factory settings.")
        settings = payload["settings"]
        if int(settings["benchmark_workers"]) < 1 or float(settings["base_capacity"]) <= 0:
            raise ValueError("Workforce and capacity references must be positive.")
        if not 0 <= float(settings["max_overtime"]) <= 0.30:
            raise ValueError("Overtime must be between 0% and 30%.")
        master = payload_frame(payload["machine_master"])
        names = master["Process"].astype(str).str.strip()
        if master.empty or (names == "").any() or names.duplicated().any():
            raise ValueError("Enter at least one machine process, with a unique name for every process.")
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            self._replace(conn, "configuration", payload, revision, actor, now)
            conn.execute(self.config_audit.insert().values(
                revision=revision + 1, payload=dumps(payload), saved_at=now, saved_by=actor))
        return Record(copy.deepcopy(payload), revision + 1, now, actor)

    def save_plan(self, inputs, snapshot, artifacts, revision, config_revision, actor, role, save_id):
        if role not in {"Admin", "Planner"}:
            raise PermissionError("Sign in as Admin or Planner to save a plan.")
        planning_date = str(date.fromisoformat(inputs["planning_date"]))
        orders = payload_frame(inputs["orders"])
        if not orders.empty:
            ids = orders["Order"].fillna("").astype(str).str.strip()
            if (ids == "").any() or orders["Due Date"].isna().any():
                raise ValueError("Every saved order needs an ID and a valid due date.")
            validate_order_ids(ids)
            amounts = orders[["Original Quantity", "Completed Before Today", "Actual Production Today", "Workers Used Today"]]
            if amounts.isna().any().any() or (amounts < 0).any().any():
                raise ValueError("Production quantities and worker counts cannot be negative or missing.")
            if (orders["Completed Before Today"] + orders["Actual Production Today"] > orders["Original Quantity"]).any():
                raise ValueError("Completed production cannot exceed original quantity.")
        if inputs["workers_present"] < 0:
            raise ValueError("Attendance cannot be negative.")
        now = datetime.now(timezone.utc).isoformat()
        with self.engine.begin() as conn:
            prior = conn.execute(select(self.replans).where(self.replans.c.save_id == save_id)).mappings().first()
            if prior:
                prior_payload = json.loads(prior["payload"])
                if (prior_payload["inputs"] != inputs or prior["saved_by"] != actor
                        or prior_payload["summary"] != snapshot or prior_payload["artifacts"] != artifacts):
                    raise ValueError("A save identifier cannot be reused for different data.")
                return Record(inputs, prior["revision"], prior["saved_at"], actor)
            # Lock/check configuration too: the saved forecast must use its current references.
            checked = conn.execute(update(self.documents).where(
                self.documents.c.name == "configuration", self.documents.c.revision == config_revision,
            ).values(revision=config_revision))
            if checked.rowcount != 1:
                raise StorageConflict("Admin settings changed. Reload saved data and replan before saving.")
            configuration = self._read(conn, "configuration")
            current = self._read(conn, "operations")
            if planning_date < current.payload["planning_date"]:
                raise ValueError("Earlier daily records are read-only; save changes on the current production date.")
            self._replace(conn, "operations", inputs, revision, actor, now)
            saved = {"inputs": inputs, "configuration": configuration.payload, "summary": snapshot, "artifacts": artifacts}
            statement = self.insert(self.days).values(planning_date=planning_date,
                payload=dumps(saved), updated_at=now, updated_by=actor)
            conn.execute(statement.on_conflict_do_update(index_elements=["planning_date"], set_={
                "payload": statement.excluded.payload, "updated_at": now, "updated_by": actor}))
            conn.execute(self.replans.insert().values(save_id=save_id, revision=revision + 1,
                planning_date=planning_date, payload=dumps(saved), saved_at=now, saved_by=actor))
        return Record(copy.deepcopy(inputs), revision + 1, now, actor)

    def history(self, limit=None):
        query = select(self.replans).order_by(self.replans.c.revision.desc())
        if limit is not None:
            query = query.limit(limit)
        with self.engine.connect() as conn:
            rows = list(conn.execute(query).mappings())
        return [{**json.loads(row["payload"])["summary"], "save_id": row["save_id"],
                 "revision": row["revision"], "saved_at": row["saved_at"], "saved_by": row["saved_by"]}
                for row in reversed(rows)]

    def daily_record(self, planning_date):
        with self.engine.connect() as conn:
            row = conn.execute(select(self.days).where(self.days.c.planning_date == str(planning_date))).mappings().first()
        return json.loads(row["payload"]) if row else None

    def daily_dates(self):
        with self.engine.connect() as conn:
            return list(conn.execute(select(self.days.c.planning_date).order_by(self.days.c.planning_date.desc())).scalars())

    def export_records(self):
        """Portable data backup without connection strings or account credentials."""
        with self.engine.connect() as conn:
            return {"schema_version": 1, "exported_at": datetime.now(timezone.utc).isoformat(),
                    "tables": {table.name: [dict(row) for row in conn.execute(select(table)).mappings()]
                               for table in (self.documents, self.days, self.replans, self.config_audit)}}
