"""Session drafts over shared saved data; no scheduling or ML calculations."""
import copy
from datetime import date
from uuid import uuid4

from rapid_storage import (
    StorageConflict, configuration_payload, operations_payload, payload_frame,
    dumps, next_day_payload,
)


class Workspace:
    def __init__(self, store, state, actor, role):
        self.store, self.state, self.actor, self.role = store, state, actor, role

    def _apply_operations(self, payload):
        self.state["planning_date"] = date.fromisoformat(payload["planning_date"])
        self.state["workers_present"] = int(payload["workers_present"])
        self.state["orders"] = payload_frame(payload["orders"])
        self.state["machine_today"] = payload_frame(payload["machine_today"])
        self.state["production_notes"] = copy.deepcopy(payload.get("production_notes", []))
        self.state["actions"] = copy.deepcopy(payload.get("actions", []))
        self.state["_base_operations"] = operations_payload(self.state)
        self.state["_widget_epoch"] = self.state.get("_widget_epoch", 0) + 1

    def reload(self):
        config, ops = self.store.load()
        self.state["settings"] = copy.deepcopy(config.payload["settings"])
        self.state["machine_master"] = payload_frame(config.payload["machine_master"])
        self.state["_config_revision"] = config.revision
        self.state["_base_configuration"] = copy.deepcopy(config.payload)
        self.state["_ops_revision"] = ops.revision
        self.state["_latest_date"] = ops.payload["planning_date"]
        self.state["_historical"] = False
        self.state["_saved_at"] = ops.updated_at
        self._apply_operations(ops.payload)
        self.state["plan_history"] = self.store.history(limit=90)
        self.state["_storage_loaded"] = True
        self.state.pop("_pending_save", None)

    def configuration_changed(self):
        return dumps(configuration_payload(self.state)) != dumps(self.state["_base_configuration"])

    def operations_changed(self):
        return dumps(operations_payload(self.state)) != dumps(self.state["_base_operations"])

    def save_configuration(self):
        record = self.store.save_configuration(configuration_payload(self.state),
            self.state["_config_revision"], self.actor, self.role)
        self.state["_config_revision"] = record.revision
        self.state["_base_configuration"] = record.payload
        return record

    def switch_date(self, requested):
        if (not self.state.get("_historical") and self.operations_changed()) or self.configuration_changed():
            raise ValueError("Save your current changes, or reload saved data to discard the draft, before changing the planning date.")
        _, latest = self.store.load()
        if latest.revision != self.state["_ops_revision"]:
            raise StorageConflict("Another planner updated the plan. Reload saved data before changing dates.")
        requested = str(requested)
        if requested < latest.payload["planning_date"]:
            saved_day = self.store.daily_record(requested)
            if saved_day is None:
                raise ValueError("No saved daily record exists for this date.")
            self._apply_operations(saved_day["inputs"])
            self.state["_historical"] = True
        else:
            self._apply_operations(next_day_payload(latest.payload, requested))
            self.state["_historical"] = False

    def save_plan(self, snapshot, artifacts):
        if self.state.get("_historical"):
            raise ValueError("Historical records are read-only.")
        if self.configuration_changed():
            raise ValueError("Save Admin Settings first so the saved plan uses the same reference values.")
        inputs = operations_payload(self.state)
        signature = dumps({"inputs": inputs, "snapshot": snapshot, "artifacts": artifacts})
        pending = self.state.get("_pending_save")
        if not pending or pending["signature"] != signature:
            pending = {"signature": signature, "id": str(uuid4())}
            self.state["_pending_save"] = pending
        record = self.store.save_plan(inputs, snapshot, artifacts, self.state["_ops_revision"],
            self.state["_config_revision"], self.actor, self.role, pending["id"])
        self.state["_ops_revision"] = record.revision
        self.state["_base_operations"] = copy.deepcopy(inputs)
        self.state["_latest_date"] = inputs["planning_date"]
        self.state["_saved_at"] = record.updated_at
        self.state["plan_history"] = self.store.history(limit=90)
        self.state.pop("_pending_save", None)
        return record
