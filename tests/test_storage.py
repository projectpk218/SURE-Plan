"""Persistence regressions against independent connections to a real SQLite file."""
import copy
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from rapid_storage import (
    RapidStore,
    StorageConflict,
    configuration_payload,
    dumps,
    frame_payload,
    next_day_payload,
    operations_payload,
    payload_frame,
)


def sample_state():
    return {
        "planning_date": date(2026, 9, 10),
        "workers_present": 120,
        "settings": {"benchmark_workers": 150, "base_capacity": 300, "max_overtime": 0.20},
        "machine_master": pd.DataFrame([
            {"Process": "Stitching", "Total Machines": 12},
            {"Process": "Cutting", "Total Machines": 8},
        ]),
        "machine_today": pd.DataFrame([
            {"Process": "Stitching", "Available Machines": 10},
            {"Process": "Cutting", "Available Machines": 8},
        ]),
        "orders": pd.DataFrame([
            {"Order": "A", "Original Quantity": 1000, "Completed Before Today": 100,
             "Actual Production Today": 35, "Workers Used Today": 20,
             "Due Date": pd.Timestamp("2026-09-15"),
             "Expected Material Ready Date": pd.NaT, "Material Status": "Ready"},
            {"Order": "B", "Original Quantity": 2000, "Completed Before Today": 200,
             "Actual Production Today": 0, "Workers Used Today": 0,
             "Due Date": pd.Timestamp("2026-09-18"),
             "Expected Material Ready Date": pd.Timestamp("2026-09-14"),
             "Material Status": "Awaiting Material"},
        ]),
    }


class StorageRegressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="rapid-storage-test-")
        self.url = "sqlite:///" + (Path(self.temp.name) / "rapid.sqlite3").as_posix()
        self.stores = []
        self.store = self.open_store()
        state = sample_state()
        self.config = configuration_payload(state)
        self.inputs = operations_payload(state)
        self.summary = {"planning_date": "2026-09-10", "late_orders": 1, "total_delay": 4}
        self.artifacts = {"results": frame_payload(pd.DataFrame([
            {"Order": "A", "Completion Date": pd.NaT, "On Time": "UNKNOWN"},
        ])), "daily": frame_payload(pd.DataFrame([{"day": 1, "workers": 120}]))}
        self.store.initialize(self.config, self.inputs)

    def tearDown(self):
        for store in self.stores:
            store.close()
        self.temp.cleanup()

    def open_store(self):
        store = RapidStore(self.url)
        self.stores.append(store)
        return store

    def save(self, *, store=None, inputs=None, summary=None, artifacts=None,
             revision=0, config_revision=0, actor="planner", role="Planner", save_id="save-1"):
        return (store or self.store).save_plan(
            self.inputs if inputs is None else inputs,
            self.summary if summary is None else summary,
            self.artifacts if artifacts is None else artifacts,
            revision, config_revision, actor, role, save_id,
        )

    def test_reopen_preserves_inputs_configuration_and_complete_daily_record(self):
        self.save()
        self.store.close()
        reopened = self.open_store()
        config, operations = reopened.load()
        self.assertEqual(config.payload, self.config)
        self.assertEqual(operations.payload, self.inputs)
        self.assertEqual(operations.revision, 1)
        self.assertEqual(operations.updated_by, "planner")
        self.assertEqual(reopened.daily_record("2026-09-10"), {
            "inputs": self.inputs, "configuration": self.config,
            "summary": self.summary, "artifacts": self.artifacts,
        })

    def test_initialization_does_not_overwrite_existing_records(self):
        self.save()
        different = copy.deepcopy(self.inputs)
        different["workers_present"] = 1
        self.open_store().initialize({"unexpected": "defaults"}, different)
        configuration, operations = self.store.load()
        self.assertEqual(configuration.payload, self.config)
        self.assertEqual(operations.payload, self.inputs)
        self.assertEqual(operations.revision, 1)

    def test_configuration_requires_admin_and_creates_revision_audit(self):
        changed = copy.deepcopy(self.config)
        changed["settings"]["base_capacity"] = 350
        for role in ("Planner", "Guest", "", "admin"):
            with self.subTest(role=role), self.assertRaises(PermissionError):
                self.store.save_configuration(changed, 0, "person", role)
        self.assertEqual(self.store.load()[0].revision, 0)
        saved = self.store.save_configuration(changed, 0, "factory admin", "Admin")
        self.assertEqual(saved.revision, 1)
        self.assertEqual(self.open_store().load()[0].payload, changed)
        audit = self.store.export_records()["tables"]["rapid_configuration_history"]
        self.assertEqual(len(audit), 1)
        self.assertEqual(audit[0]["saved_by"], "factory admin")

    def test_stale_admin_cannot_overwrite_newer_configuration(self):
        second = self.open_store()
        stale_configuration = second.load()[0]
        self.store.save_configuration(self.config, 0, "first admin", "Admin")
        with self.assertRaises(StorageConflict):
            second.save_configuration(self.config, stale_configuration.revision, "second admin", "Admin")
        self.assertEqual(self.store.load()[0].updated_by, "first admin")
        self.assertEqual(len(self.store.export_records()["tables"]["rapid_configuration_history"]), 1)

    def test_two_sessions_cannot_lose_an_operations_update(self):
        second = self.open_store()
        first_revision = self.store.load()[1].revision
        second_revision = second.load()[1].revision
        first_inputs = copy.deepcopy(self.inputs)
        first_inputs["workers_present"] = 110
        self.save(inputs=first_inputs, revision=first_revision, actor="first planner")
        with self.assertRaises(StorageConflict):
            self.save(store=second, revision=second_revision, actor="second planner", save_id="save-2")
        self.assertEqual(self.store.load()[1].payload, first_inputs)
        self.assertEqual(self.store.daily_record("2026-09-10")["inputs"], first_inputs)
        self.assertEqual(len(self.store.history()), 1)

    def test_stale_configuration_guard_rolls_back_entire_plan_save(self):
        self.save()
        second = self.open_store()
        changed = copy.deepcopy(self.config)
        changed["settings"]["base_capacity"] = 450
        second.save_configuration(changed, 0, "admin", "Admin")
        before = self.store.daily_record("2026-09-10")
        with self.assertRaisesRegex(StorageConflict, "Admin settings changed"):
            self.save(revision=1, config_revision=0, save_id="stale-configuration")
        self.assertEqual(self.store.load()[1].revision, 1)
        self.assertEqual(self.store.daily_record("2026-09-10"), before)
        self.assertEqual(len(self.store.history()), 1)
        self.save(revision=1, config_revision=1, save_id="fresh-configuration")
        self.assertEqual(self.store.daily_record("2026-09-10")["configuration"], changed)

    def test_save_token_is_idempotent_after_connection_reopens(self):
        first = self.save()
        retry = self.save(store=self.open_store())
        self.assertEqual(retry, first)
        self.assertEqual(self.store.load()[1].revision, 1)
        self.assertEqual(len(self.store.history()), 1)
        self.assertEqual(self.store.daily_dates(), ["2026-09-10"])

    def test_save_token_cannot_be_reused_for_different_inputs_or_actor(self):
        self.save()
        changed = copy.deepcopy(self.inputs)
        changed["workers_present"] = 100
        with self.assertRaises(ValueError):
            self.save(inputs=changed)
        with self.assertRaises(ValueError):
            self.save(actor="different planner")
        self.assertEqual(self.store.load()[1].revision, 1)

    def test_save_token_cannot_be_reused_for_different_summary(self):
        self.save()
        changed = {**self.summary, "late_orders": 99}
        with self.assertRaises(ValueError):
            self.save(summary=changed)
        self.assertEqual(self.store.daily_record("2026-09-10")["summary"], self.summary)
        self.assertEqual(self.store.load()[1].revision, 1)

    def test_save_token_cannot_be_reused_for_different_forecast_artifacts(self):
        self.save()
        changed = copy.deepcopy(self.artifacts)
        changed["daily"] = frame_payload(pd.DataFrame([{"day": 1, "workers": 80}]))
        with self.assertRaises(ValueError):
            self.save(artifacts=changed)
        self.assertEqual(self.store.daily_record("2026-09-10")["artifacts"], self.artifacts)
        self.assertEqual(self.store.load()[1].revision, 1)

    def test_daily_record_is_latest_snapshot_but_each_replan_is_retained(self):
        self.save()
        changed = copy.deepcopy(self.inputs)
        changed["workers_present"] = 130
        new_summary = {**self.summary, "late_orders": 0}
        self.save(inputs=changed, summary=new_summary, revision=1, save_id="save-2")
        history = self.store.history()
        self.assertEqual([row["late_orders"] for row in history], [1, 0])
        self.assertEqual([row["revision"] for row in history], [1, 2])
        self.assertEqual(self.store.daily_record("2026-09-10")["inputs"], changed)
        self.assertEqual(self.store.daily_record("2026-09-10")["summary"], new_summary)

    def test_history_and_daily_records_are_not_truncated_at_90(self):
        start = date(2026, 9, 10)
        for number in range(95):
            day = str(start + timedelta(days=number))
            inputs = {**self.inputs, "planning_date": day}
            self.save(inputs=inputs, summary={**self.summary, "planning_date": day, "sequence": number},
                      revision=number, save_id="save-" + str(number))
        reopened = self.open_store()
        self.assertEqual(len(reopened.history()), 95)
        self.assertEqual([row["sequence"] for row in reopened.history()], list(range(95)))
        self.assertEqual(len(reopened.daily_dates()), 95)
        self.assertIsNotNone(reopened.daily_record(start))
        self.assertEqual([row["sequence"] for row in reopened.history(limit=3)], [92, 93, 94])

    def test_rollover_adds_actuals_once_and_clears_daily_entries(self):
        original = copy.deepcopy(self.inputs)
        tomorrow = next_day_payload(original, "2026-09-11")
        rows = payload_frame(tomorrow["orders"])
        self.assertEqual(rows["Completed Before Today"].tolist(), [135, 200])
        self.assertEqual(rows["Actual Production Today"].tolist(), [0, 0])
        self.assertEqual(rows["Workers Used Today"].tolist(), [0, 0])
        self.assertEqual(next_day_payload(tomorrow, "2026-09-11"), tomorrow)
        following = next_day_payload(tomorrow, "2026-09-14")
        self.assertEqual(payload_frame(following["orders"])["Completed Before Today"].tolist(), [135, 200])
        self.assertEqual(original, self.inputs, "Rollover must not mutate the saved source record")
        self.assertEqual(tomorrow["workers_present"], original["workers_present"])
        self.assertEqual(tomorrow["machine_today"], original["machine_today"])

    def test_rollover_reopened_store_keeps_yesterday_and_does_not_double_count(self):
        self.save()
        tomorrow = next_day_payload(self.store.load()[1].payload, "2026-09-11")
        self.save(inputs=tomorrow, revision=1, save_id="tomorrow")
        reopened = self.open_store()
        next_open = next_day_payload(reopened.load()[1].payload, "2026-09-11")
        self.assertEqual(payload_frame(next_open["orders"])["Completed Before Today"].tolist(), [135, 200])
        yesterday = reopened.daily_record("2026-09-10")
        self.assertEqual(payload_frame(yesterday["inputs"]["orders"])["Actual Production Today"].tolist(), [35, 0])
        self.assertEqual(reopened.daily_dates(), ["2026-09-11", "2026-09-10"])

    def test_same_day_rollover_preserves_actuals(self):
        result = next_day_payload(self.inputs, "2026-09-10")
        self.assertEqual(result, self.inputs)
        result["workers_present"] = 1
        self.assertEqual(self.inputs["workers_present"], 120)

    def test_earlier_date_rollover_and_save_are_rejected_without_changes(self):
        with self.assertRaises(ValueError):
            next_day_payload(self.inputs, "2026-09-09")
        self.save()
        past = {**self.inputs, "planning_date": "2026-09-09"}
        with self.assertRaisesRegex(ValueError, "Earlier daily records"):
            self.save(inputs=past, revision=1, save_id="past")
        self.assertEqual(self.store.load()[1].revision, 1)
        self.assertIsNone(self.store.daily_record("2026-09-09"))
        self.assertEqual(len(self.store.history()), 1)

    def test_date_and_missing_date_serialization_round_trip(self):
        decoded = payload_frame(self.inputs["orders"])
        self.assertEqual(decoded.iloc[0]["Due Date"], pd.Timestamp("2026-09-15"))
        self.assertTrue(pd.isna(decoded.iloc[0]["Expected Material Ready Date"]))
        self.assertEqual(frame_payload(decoded), self.inputs["orders"])
        self.save()
        results = payload_frame(self.store.daily_record("2026-09-10")["artifacts"]["results"])
        self.assertTrue(pd.isna(results.iloc[0]["Completion Date"]))
        self.assertEqual(results.iloc[0]["On Time"], "UNKNOWN")
        self.assertIn("null", dumps(self.artifacts))

    def test_failure_after_operations_write_rolls_back_revision_and_history(self):
        self.save()
        changed = {**self.inputs, "workers_present": 100}
        invalid_summary = {"unsupported": {"a Python set is not JSON"}}
        with self.assertRaises(TypeError):
            self.save(inputs=changed, summary=invalid_summary, revision=1, save_id="failing-save")
        self.assertEqual(self.store.load()[1].revision, 1)
        self.assertEqual(self.store.load()[1].payload, self.inputs)
        self.assertEqual(self.store.daily_record("2026-09-10")["inputs"], self.inputs)
        self.assertEqual(len(self.store.history()), 1)
        self.save(inputs=changed, revision=1, save_id="failing-save")
        self.assertEqual(self.store.load()[1].revision, 2)

    def test_unauthorized_plan_save_is_rejected(self):
        for role in ("Guest", "", "viewer"):
            with self.subTest(role=role), self.assertRaises(PermissionError):
                self.save(role=role)
        self.assertEqual(self.store.load()[1].revision, 0)
        self.assertEqual(self.store.history(), [])

    def test_invalid_order_inputs_are_rejected_without_any_persistence(self):
        for column, value in [("Order", ""), ("Due Date", pd.NaT),
                              ("Actual Production Today", -1), ("Actual Production Today", 1001),
                              ("Workers Used Today", None)]:
            with self.subTest(column=column, value=value):
                inputs = copy.deepcopy(self.inputs)
                rows = payload_frame(inputs["orders"])
                rows.loc[0, column] = value
                inputs["orders"] = frame_payload(rows)
                with self.assertRaises(ValueError):
                    self.save(inputs=inputs)
        self.assertEqual(self.store.load()[1].revision, 0)
        self.assertEqual(self.store.history(), [])

    def test_duplicate_order_ids_are_rejected(self):
        inputs = copy.deepcopy(self.inputs)
        rows = payload_frame(inputs["orders"])
        rows.loc[1, "Order"] = " A "
        inputs["orders"] = frame_payload(rows)
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            self.save(inputs=inputs)
        self.assertEqual(self.store.load()[1].revision, 0)


if __name__ == "__main__":
    unittest.main()
