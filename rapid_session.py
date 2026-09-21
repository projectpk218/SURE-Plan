"""Private, temporary workspace for one Streamlit session; no database connection."""
import copy
from datetime import datetime, timezone

from rapid_storage import Record, StorageConflict, dumps, payload_frame
from planner_core import validate_order_ids


class SessionStore:
    is_local = False

    def __init__(self):
        self.documents = {}
        self.days = {}
        self.replans = {}
        self.configuration_history = []

    def initialize(self, configuration, operations):
        if not self.documents:
            now = datetime.now(timezone.utc).isoformat()
            self.documents = {name: Record(copy.deepcopy(payload), 0, now, "session")
                              for name, payload in [("configuration", configuration), ("operations", operations)]}

    def load(self):
        return copy.deepcopy((self.documents["configuration"], self.documents["operations"]))

    def save_configuration(self, payload, revision, actor, role):
        if role != "Admin":
            raise PermissionError("Only Admin can save factory settings.")
        if self.documents["configuration"].revision != revision:
            raise StorageConflict("Settings changed. Reload the saved session.")
        settings = payload["settings"]
        if settings["benchmark_workers"] < 1 or settings["base_capacity"] <= 0 or not 0 <= settings["max_overtime"] <= .3:
            raise ValueError("Enter valid workforce, capacity and overtime references.")
        master = payload_frame(payload["machine_master"])
        names = master["Process"].astype(str).str.strip()
        if master.empty or names.eq("").any() or names.duplicated().any():
            raise ValueError("Enter unique machine process names.")
        dumps(payload)
        record = Record(copy.deepcopy(payload), revision + 1, datetime.now(timezone.utc).isoformat(), actor)
        self.documents["configuration"] = record
        self.configuration_history.append(copy.deepcopy(record.__dict__))
        return copy.deepcopy(record)

    def save_plan(self, inputs, snapshot, artifacts, revision, config_revision, actor, role, save_id):
        if role not in {"Admin", "Planner"}:
            raise PermissionError("Sign in to save a plan.")
        prior = self.replans.get(save_id)
        if prior:
            if prior["inputs"] != inputs or prior["summary"] != snapshot or prior["artifacts"] != artifacts or prior["saved_by"] != actor:
                raise ValueError("A save identifier cannot be reused for different data.")
            return Record(copy.deepcopy(inputs), prior["revision"], prior["saved_at"], actor)
        config, current = self.load()
        if config.revision != config_revision or current.revision != revision:
            raise StorageConflict("The saved session changed. Reload before saving.")
        if inputs["planning_date"] < current.payload["planning_date"]:
            raise ValueError("Earlier daily records are read-only.")
        orders = payload_frame(inputs["orders"])
        if not orders.empty:
            ids = orders["Order"].fillna("").astype(str).str.strip()
            validate_order_ids(ids)
            if ids.eq("").any() or orders["Due Date"].isna().any():
                raise ValueError("Every saved order needs an ID and valid due date.")
            amounts = orders[["Original Quantity", "Completed Before Today", "Actual Production Today", "Workers Used Today"]]
            if amounts.isna().any().any() or (amounts < 0).any().any():
                raise ValueError("Production and worker values cannot be negative or missing.")
            if (orders["Completed Before Today"] + orders["Actual Production Today"] > orders["Original Quantity"]).any():
                raise ValueError("Completed production cannot exceed original quantity.")
        if inputs["workers_present"] < 0:
            raise ValueError("Attendance cannot be negative.")
        now = datetime.now(timezone.utc).isoformat()
        saved = copy.deepcopy({"inputs": inputs, "configuration": config.payload, "summary": snapshot, "artifacts": artifacts,
                               "save_id": save_id, "revision": revision + 1, "saved_at": now, "saved_by": actor})
        dumps(saved)
        self.days[inputs["planning_date"]] = saved
        self.replans[save_id] = saved
        self.documents["operations"] = Record(copy.deepcopy(inputs), revision + 1, now, actor)
        return copy.deepcopy(self.documents["operations"])

    def history(self, limit=None):
        rows = list(self.replans.values())
        if limit is not None:
            rows = rows[-limit:] if limit else []
        return copy.deepcopy([{**row["summary"], **{k: row[k] for k in ("save_id", "revision", "saved_at", "saved_by")}} for row in rows])

    def daily_dates(self):
        return sorted(self.days, reverse=True)

    def daily_record(self, planning_date):
        return copy.deepcopy(self.days.get(str(planning_date)))

    def export_records(self):
        return copy.deepcopy({"storage_mode": "temporary_session", "documents": {k: v.__dict__ for k, v in self.documents.items()},
                              "daily_records": self.days, "replans": self.replans, "configuration_history": self.configuration_history})
