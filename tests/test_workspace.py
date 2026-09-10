"""Shared-record workflows with separate browser-session state dictionaries."""
import copy
from datetime import date
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rapid_storage import RapidStore, StorageConflict, configuration_payload, operations_payload
from rapid_workspace import Workspace
from test_storage import sample_state


class WorkspaceRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rapid-workspace-test-")
        url = "sqlite:///" + (Path(self.temp.name) / "workspace.sqlite3").as_posix()
        self.store = RapidStore(url)
        self.second_store = RapidStore(url)
        sample = sample_state()
        self.original_inputs = operations_payload(sample)
        self.store.initialize(configuration_payload(sample), self.original_inputs)
        self.state = {}
        self.workspace = Workspace(self.store, self.state, "planner one", "Planner")
        self.workspace.reload()
        self.other_state = {}
        self.other = Workspace(self.second_store, self.other_state, "admin two", "Admin")
        self.other.reload()
        self.summary = {"planning_date": "2026-09-10", "late_orders": 1}
        self.artifacts = {"report": {"columns": ["Order", "Delay"], "data": [["A", 1]]}}

    def tearDown(self):
        self.store.close()
        self.second_store.close()
        self.temp.cleanup()

    def save(self, workspace=None):
        target = workspace or self.workspace
        return target.save_plan({**self.summary, "planning_date": str(target.state["planning_date"])},
                                copy.deepcopy(self.artifacts))

    def prepare_two_days(self):
        self.save()
        self.workspace.switch_date(date(2026, 9, 11))
        self.save()
        self.workspace.reload()

    def test_reload_populates_shared_state_without_shared_mutable_objects(self):
        self.assertEqual(self.state["settings"], self.other_state["settings"])
        self.assertTrue(self.state["orders"].equals(self.other_state["orders"]))
        self.assertFalse(self.workspace.configuration_changed())
        self.assertFalse(self.workspace.operations_changed())
        self.state["settings"]["base_capacity"] = 999
        self.state["orders"].loc[0, "Actual Production Today"] = 50
        self.assertEqual(self.other_state["settings"]["base_capacity"], 300)
        self.assertEqual(self.other_state["orders"].loc[0, "Actual Production Today"], 35)
        self.assertEqual(self.store.load()[1].payload, self.original_inputs)

    def test_admin_saved_settings_are_shared_after_planner_reload(self):
        self.other_state["settings"]["base_capacity"] = 360
        self.other_state["machine_master"].loc[0, "Total Machines"] = 15
        self.other.save_configuration()
        self.assertFalse(self.other.configuration_changed())
        self.assertEqual(self.state["settings"]["base_capacity"], 300)
        self.workspace.reload()
        self.assertEqual(self.state["settings"]["base_capacity"], 360)
        self.assertEqual(self.state["machine_master"].loc[0, "Total Machines"], 15)
        self.assertEqual(self.state["_config_revision"], 1)
        self.assertFalse(self.workspace.configuration_changed())

    def test_planner_cannot_save_admin_configuration(self):
        self.state["settings"]["base_capacity"] = 360
        with self.assertRaises(PermissionError):
            self.workspace.save_configuration()
        self.assertEqual(self.store.load()[0].revision, 0)
        self.assertTrue(self.workspace.configuration_changed())

    def test_unsaved_configuration_prevents_plan_save(self):
        self.other_state["settings"]["base_capacity"] = 360
        with self.assertRaisesRegex(ValueError, "Save Admin Settings first"):
            self.save(self.other)
        self.assertEqual(self.store.load()[1].revision, 0)
        self.assertEqual(self.store.history(), [])
        self.other.save_configuration()
        self.save(self.other)
        saved = self.store.daily_record("2026-09-10")
        self.assertEqual(saved["configuration"]["settings"]["base_capacity"], 360)

    def test_unsaved_operational_draft_prevents_date_switch(self):
        self.state["orders"].loc[0, "Actual Production Today"] = 40
        with self.assertRaisesRegex(ValueError, "Save your current changes"):
            self.workspace.switch_date(date(2026, 9, 11))
        self.assertEqual(self.state["planning_date"], date(2026, 9, 10))
        self.assertEqual(self.state["orders"].loc[0, "Actual Production Today"], 40)
        self.assertEqual(self.store.load()[1].revision, 0)

    def test_unsaved_configuration_prevents_date_switch(self):
        self.state["settings"]["base_capacity"] = 400
        with self.assertRaisesRegex(ValueError, "Save your current changes"):
            self.workspace.switch_date(date(2026, 9, 11))
        self.assertEqual(self.state["planning_date"], date(2026, 9, 10))

    def test_new_day_rollover_resets_daily_values_and_carries_completed_once(self):
        self.save()
        self.workspace.switch_date(date(2026, 9, 11))
        self.assertEqual(self.state["orders"]["Completed Before Today"].tolist(), [135, 200])
        self.assertEqual(self.state["orders"]["Actual Production Today"].tolist(), [0, 0])
        self.assertEqual(self.state["orders"]["Workers Used Today"].tolist(), [0, 0])
        self.assertEqual(self.state["workers_present"], 120)
        self.assertFalse(self.state["_historical"])
        self.workspace.switch_date(date(2026, 9, 11))
        self.assertEqual(self.state["orders"]["Completed Before Today"].tolist(), [135, 200])
        self.save()
        self.other.reload()
        self.assertEqual(self.other_state["planning_date"], date(2026, 9, 11))
        self.assertEqual(self.other_state["orders"]["Completed Before Today"].tolist(), [135, 200])
        self.assertEqual(self.other_state["orders"]["Actual Production Today"].tolist(), [0, 0])

    def test_historical_date_is_read_only_and_can_return_to_latest(self):
        self.prepare_two_days()
        self.workspace.switch_date(date(2026, 9, 10))
        self.assertTrue(self.state["_historical"])
        self.assertEqual(self.state["orders"]["Actual Production Today"].tolist(), [35, 0])
        with self.assertRaisesRegex(ValueError, "Historical records are read-only"):
            self.save()
        self.workspace.switch_date(date(2026, 9, 11))
        self.assertFalse(self.state["_historical"])
        self.assertEqual(self.state["orders"]["Actual Production Today"].tolist(), [0, 0])
        self.assertEqual(self.state["orders"]["Completed Before Today"].tolist(), [135, 200])
        self.assertEqual(self.store.load()[1].revision, 2)

    def test_historical_widget_sync_cannot_trap_user_on_read_only_date(self):
        self.prepare_two_days()
        self.workspace.switch_date(date(2026, 9, 10))
        # Streamlit widget synchronization can reassign historical display values;
        # an unsavable historical draft must still allow returning to the latest plan.
        self.state["workers_present"] = 119
        self.assertTrue(self.workspace.operations_changed())
        self.workspace.switch_date(date(2026, 9, 11))
        self.assertFalse(self.state["_historical"])
        self.assertEqual(self.state["workers_present"], 120)
        self.assertEqual(self.store.daily_record("2026-09-10")["inputs"]["workers_present"], 120)

    def test_switch_to_missing_historical_date_preserves_current_state(self):
        before = operations_payload(self.state)
        with self.assertRaisesRegex(ValueError, "No saved daily record"):
            self.workspace.switch_date(date(2026, 9, 9))
        self.assertEqual(operations_payload(self.state), before)
        self.assertFalse(self.state["_historical"])

    def test_separate_session_save_requires_reload_before_switching_dates(self):
        self.other_state["workers_present"] = 100
        self.save(self.other)
        with self.assertRaises(StorageConflict):
            self.workspace.switch_date(date(2026, 9, 11))
        self.workspace.reload()
        self.workspace.switch_date(date(2026, 9, 11))
        self.assertEqual(self.state["workers_present"], 100)

    def test_reload_discards_draft_and_pending_save_and_restores_latest_date(self):
        self.prepare_two_days()
        self.workspace.switch_date(date(2026, 9, 10))
        self.state["workers_present"] = 1
        self.state["settings"]["base_capacity"] = 1
        self.state["_pending_save"] = {"id": "incomplete", "signature": "changed"}
        epoch = self.state["_widget_epoch"]
        self.workspace.reload()
        self.assertEqual(self.state["planning_date"], date(2026, 9, 11))
        self.assertEqual(self.state["workers_present"], 120)
        self.assertEqual(self.state["settings"]["base_capacity"], 300)
        self.assertNotIn("_pending_save", self.state)
        self.assertFalse(self.state["_historical"])
        self.assertFalse(self.workspace.operations_changed())
        self.assertFalse(self.workspace.configuration_changed())
        self.assertGreater(self.state["_widget_epoch"], epoch)

    def test_retry_after_uncertain_save_reuses_token_without_duplicate_history(self):
        real_save = self.store.save_plan

        def save_then_lose_response(*args, **kwargs):
            real_save(*args, **kwargs)
            raise ConnectionError("response lost after committed transaction")

        with patch.object(self.store, "save_plan", side_effect=save_then_lose_response):
            with self.assertRaises(ConnectionError):
                self.save()
        pending_id = self.state["_pending_save"]["id"]
        self.assertEqual(self.state["_ops_revision"], 0)
        self.assertEqual(len(self.store.history()), 1)
        retried = self.save()
        self.assertEqual(retried.revision, 1)
        self.assertEqual(self.state["_ops_revision"], 1)
        self.assertNotIn("_pending_save", self.state)
        self.assertEqual([row["save_id"] for row in self.store.history()], [pending_id])


if __name__ == "__main__":
    unittest.main()
