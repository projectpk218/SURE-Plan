"""Daily targets come from the plan, stay fixed once accepted, and isolate dates."""
from copy import deepcopy
from datetime import date
import unittest
import pandas as pd

from production_management import planned_daily_notes, daily_progress


class DailyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2026, 10, 8)
        self.orders = pd.DataFrame([
            {"Order": "A", "Original Quantity": 1000, "Completed Before Today": 200,
             "Actual Production Today": 150, "Current Process": "Cutting"},
            {"Order": "B", "Original Quantity": 500, "Completed Before Today": 0,
             "Actual Production Today": 0, "Current Process": "Sewing"},
        ])
        self.daily = pd.DataFrame([
            {"order": "A", "production_date": self.day, "produced": 200.8},
            {"order": "B", "production_date": self.day, "produced": 0},
            {"order": "A", "production_date": date(2026, 10, 9), "produced": 400},
        ])

    def test_target_and_progress_use_selected_day_only(self):
        notes = planned_daily_notes(self.orders, [], self.day, self.daily)
        progress = daily_progress(self.orders, notes, self.day)
        self.assertEqual(progress.loc[0, "Daily Target"], 200)
        self.assertEqual(progress.loc[0, "Target Remaining"], 50)
        self.assertEqual(progress.loc[0, "Target Reached (%)"], 75)
        self.assertEqual(progress.loc[1, "Daily Target"], 0)
        self.assertEqual(progress.loc[1, "Progress"], "No output planned")
        self.assertTrue(pd.isna(progress.loc[1, "Target Reached (%)"]))

    def test_accepted_target_does_not_move_with_actuals_or_replanning(self):
        notes = planned_daily_notes(self.orders, [], self.day, self.daily)
        original = deepcopy(notes)
        self.orders.loc[0, "Actual Production Today"] = 220
        self.daily.loc[0, "produced"] = 60
        updated = planned_daily_notes(self.orders, notes, self.day, self.daily)
        self.assertEqual(updated, original)
        progress = daily_progress(self.orders, updated, self.day)
        self.assertEqual(progress.loc[0, "Target Remaining"], 0)
        self.assertEqual(progress.loc[0, "Target Reached (%)"], 110)

    def test_manual_target_and_recorded_reason_are_preserved(self):
        notes = [{"planning_date": str(self.day), "order": "A", "daily_target": 175,
                  "recorded_reason": "Approved supervisor target"}]
        before = deepcopy(notes)
        updated = planned_daily_notes(self.orders, notes, self.day, self.daily)
        self.assertEqual(updated[0], before[0])
        self.assertEqual(notes, before)

    def test_missing_target_preserves_observations_and_caps_to_order_balance(self):
        notes = [{"planning_date": str(self.day), "order": "A", "daily_target": None,
                  "recorded_reason": "Material arrived late", "downtime_minutes": 20}]
        self.daily.loc[0, "produced"] = 900
        updated = planned_daily_notes(self.orders, notes, self.day, self.daily)
        self.assertEqual(updated[0]["daily_target"], 800)
        self.assertEqual(updated[0]["recorded_reason"], "Material arrived late")
        self.assertEqual(updated[0]["downtime_minutes"], 20)

    def test_weekend_does_not_borrow_monday_output(self):
        notes = planned_daily_notes(self.orders, [], date(2026, 10, 10), self.daily)
        self.assertTrue(all(n["daily_target"] == 0 for n in notes))

    def test_next_date_gets_new_targets_and_does_not_modify_orders(self):
        before = self.orders.copy(deep=True)
        notes = planned_daily_notes(self.orders, [], self.day, self.daily)
        updated = planned_daily_notes(self.orders, notes, date(2026, 10, 9), self.daily)
        self.assertEqual(updated[2]["daily_target"], 400)
        self.assertEqual(len(notes), 2)
        pd.testing.assert_frame_equal(self.orders, before)

    def test_empty_schedule_gives_zero_not_a_fabricated_target(self):
        notes = planned_daily_notes(self.orders, [], self.day, pd.DataFrame())
        self.assertTrue(all(n["daily_target"] == 0 for n in notes))


if __name__ == "__main__":
    unittest.main()
