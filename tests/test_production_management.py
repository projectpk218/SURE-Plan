"""Checks for factual progress, immutable inputs, and accountable action records."""
from copy import deepcopy
from datetime import date
import unittest

import pandas as pd

from production_management import (
    action_frame, action_summary, daily_entry_frame, daily_progress, new_action,
    update_actions, update_daily_notes,
)


class ProductionManagementTests(unittest.TestCase):
    def setUp(self):
        self.orders = pd.DataFrame([
            {"Order": "A", "Current Process": "Stitching", "Original Quantity": 1000,
             "Completed Before Today": 250, "Actual Production Today": 210},
            {"Order": "B", "Current Process": "Cutting", "Original Quantity": 800,
             "Completed Before Today": 100, "Actual Production Today": 100},
        ])
        self.day = date(2026, 9, 10)

    def notes(self):
        entries = daily_entry_frame(self.orders, [], self.day)
        entries.loc[0, "Daily Target"] = 300
        entries.loc[0, "Downtime Minutes"] = 45
        entries.loc[0, "Rework Quantity"] = 12
        entries.loc[0, "Recorded Reason"] = "Machine stopped; maintenance ticket 42"
        return update_daily_notes([], entries, self.orders, self.day, "Supervisor", now="t1")

    def test_unset_target_does_not_invent_shortfall_or_a_cause(self):
        progress = daily_progress(self.orders, [], self.day)
        self.assertTrue(progress["Daily Target"].isna().all())
        self.assertTrue(progress["Target Remaining"].isna().all())
        self.assertTrue((progress["Progress"] == "Target not set").all())
        self.assertTrue((progress["Recorded Reason"] == "").all())

    def test_actuals_are_authoritative_and_rework_not_double_counted(self):
        before = self.orders.copy(deep=True)
        notes = self.notes()
        progress = daily_progress(self.orders, notes, self.day)
        self.assertEqual(progress.loc[0, "Good Output Today"], 210)
        self.assertEqual(progress.loc[0, "Process"], "Stitching")
        self.assertEqual(progress.loc[0, "Target Remaining"], 90)
        self.assertEqual(progress.loc[0, "Target Reached (%)"], 70)
        self.assertEqual(progress.loc[0, "Recorded Reason"], "Machine stopped; maintenance ticket 42")
        pd.testing.assert_frame_equal(self.orders, before)
        self.assertNotIn("actual", notes[0])

    def test_explicit_target_stays_fixed_when_output_and_remaining_change(self):
        notes = self.notes()
        self.orders.loc[0, "Actual Production Today"] = 310
        progress = daily_progress(self.orders, notes, self.day)
        self.assertEqual(progress.loc[0, "Daily Target"], 300)
        self.assertEqual(progress.loc[0, "Target Remaining"], 0)
        self.assertEqual(progress.loc[0, "Progress"], "Target reached")

    def test_next_day_has_no_invented_target_or_carried_downtime(self):
        notes = self.notes()
        progress = daily_progress(self.orders, notes, date(2026, 9, 11))
        self.assertTrue(progress["Daily Target"].isna().all())
        self.assertTrue((progress["Downtime Minutes"] == 0).all())
        self.assertTrue((progress["Recorded Reason"] == "").all())
        self.assertEqual(notes[0]["planning_date"], "2026-09-10")

    def test_applying_notes_is_idempotent_and_preserves_prior_days(self):
        notes = self.notes()
        original = deepcopy(notes)
        entries = daily_entry_frame(self.orders, notes, self.day)
        self.assertEqual(update_daily_notes(notes, entries, self.orders, self.day, "Another", now="t2"), original)
        entries.loc[0, "Daily Target"] = 280
        updated = update_daily_notes(notes, entries, self.orders, date(2026, 9, 11), "Supervisor", now="t3")
        self.assertEqual(len(updated), 2)
        self.assertEqual(updated[0], original[0])
        self.assertEqual(notes, original)

    def test_blank_entries_do_not_manufacture_production_records(self):
        entries = daily_entry_frame(self.orders, [], self.day)
        self.assertEqual(update_daily_notes([], entries, self.orders, self.day, "Supervisor"), [])

    def test_invalid_values_and_duplicate_orders_rejected_without_mutation(self):
        for invalid in (-1, 2.5, float("inf"), "bad"):
            entries = daily_entry_frame(self.orders, [], self.day)
            entries["Daily Target"] = entries["Daily Target"].astype(object)
            entries.loc[0, "Daily Target"] = invalid
            with self.assertRaises(ValueError):
                update_daily_notes([], entries, self.orders, self.day, "Supervisor")
        duplicated = pd.concat([self.orders, self.orders.iloc[:1]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate order"):
            daily_progress(duplicated, [], self.day)

    def test_action_needs_an_owner_and_description(self):
        for description, owner in (("", "Maintenance"), ("Repair machine", "")):
            with self.assertRaises(ValueError):
                new_action("A", "Stitching", description, owner, self.day, self.day)

    def test_action_lifecycle_overdue_and_followups(self):
        actions = [new_action("A", "Stitching", "Inspect machine", "Maintenance", "2026-09-09", self.day,
                              "Machine stopped", actor="Planner", now="t1", action_id="a1")]
        self.assertEqual(action_summary(actions, self.day), {"open": 1, "overdue": 1, "follow_up_due": 1, "completed": 0})
        entries = action_frame(actions)
        self.assertEqual(update_actions(actions, entries, "Another", now="t2"), actions)
        entries.loc[0, "Status"] = "Done"
        updated = update_actions(actions, entries, "Maintenance", now="t2")
        self.assertEqual(updated[0]["created_at"], "t1")
        self.assertEqual(updated[0]["updated_at"], "t2")
        self.assertEqual(updated[0]["updated_by"], "Maintenance")
        self.assertEqual(actions[0]["status"], "Open")
        self.assertEqual(action_summary(updated, self.day), {"open": 0, "overdue": 0, "follow_up_due": 0, "completed": 1})

    def test_invalid_action_ids_statuses_and_dates_rejected(self):
        actions = [new_action("A", "Stitching", "Inspect machine", "Maintenance", self.day, self.day)]
        for column, value in (("Status", "Maybe"), ("Action ID", "unknown"), ("Due Date", None)):
            entries = action_frame(actions)
            entries.loc[0, column] = value
            with self.assertRaises(ValueError):
                update_actions(actions, entries, "Planner")

    def test_cancelled_actions_do_not_raise_due_alerts(self):
        actions = [new_action("Factory-wide", "Factory-wide", "Inspect machine", "Maintenance", "2026-09-01", "2026-09-01")]
        actions[0]["status"] = "Cancelled"
        self.assertEqual(action_summary(actions, self.day), {"open": 0, "overdue": 0, "follow_up_due": 0, "completed": 0})


if __name__ == "__main__":
    unittest.main()
