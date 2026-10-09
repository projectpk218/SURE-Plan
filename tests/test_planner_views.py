import unittest
from datetime import date
import pandas as pd
from rapid_views import allocation_view, confirm_staffing, delivery_view, overview_chart_data


class PlannerViewTests(unittest.TestCase):
    def setUp(self):
        self.day = date(2026, 10, 8)
        self.orders = pd.DataFrame([{"Order": "A", "Original Quantity": 100,
            "Completed Before Today": 10, "Actual Production Today": 5,
            "Workers Used Today": 0, "Due Date": self.day}])
        self.schedule = pd.DataFrame([
            {"order": "A", "current_process": "Cutting", "production_date": self.day, "workers_allocated": 3, "produced": 7.8},
            {"order": "A", "current_process": "Cutting", "production_date": date(2026, 10, 9), "workers_allocated": 9, "produced": 20},
        ])

    def test_workers_and_output_come_from_same_day(self):
        view = allocation_view(self.orders, self.schedule)
        self.assertEqual(view.loc[0, "Workers"], 3)
        self.assertEqual(view.loc[0, "Expected output"], 7)
        self.assertEqual(view.loc[0, "Order balance"], 85)

    def test_confirm_changes_only_actual_worker_records(self):
        view = allocation_view(self.orders, self.schedule)
        result = confirm_staffing(self.orders, view, 3, self.day)
        self.assertEqual(result.loc[0, "Workers Used Today"], 3)
        self.assertTrue(result.drop(columns="Workers Used Today").equals(self.orders.drop(columns="Workers Used Today")))
        self.assertEqual(self.orders.loc[0, "Workers Used Today"], 0)

    def test_cannot_confirm_future_plan_on_weekend(self):
        view = allocation_view(self.orders, self.schedule)
        with self.assertRaisesRegex(ValueError, "next working date"):
            confirm_staffing(self.orders, view, 3, date(2026, 10, 4))

    def test_overallocation_and_duplicates_are_rejected(self):
        view = allocation_view(self.orders, self.schedule)
        with self.assertRaisesRegex(ValueError, "attendance"):
            confirm_staffing(self.orders, view, 2, self.day)
        with self.assertRaises(ValueError):
            confirm_staffing(self.orders, pd.concat([view, view]), 20, self.day)

    def test_unfinished_forecast_keeps_unknown_date_and_lower_bound(self):
        results = pd.DataFrame([{"order": "A", "completion_day": None,
            "projected_delay_days": 121, "projected_delay_is_lower_bound": True, "on_time": "NO"}])
        row = delivery_view(self.orders, results, self.day).iloc[0]
        self.assertEqual(row["Forecast completion"], "Not forecast")
        self.assertEqual(row["Delay"], "At least 121 working days")
        self.assertNotIn("On schedule", row["Delivery status"])

    def test_complete_quantity_does_not_invent_completion_date(self):
        self.orders.loc[0, "Actual Production Today"] = 90
        row = delivery_view(self.orders, pd.DataFrame(), self.day).iloc[0]
        self.assertEqual(row["Delivery status"], "Quantity complete")
        self.assertEqual(row["Forecast completion"], "Completion date not recorded")

    def test_overview_charts_keep_recorded_output_and_process_staffing_distinct(self):
        progress = pd.DataFrame([
            {"Order": "A", "Daily Target": 20, "Good Output Today": 5},
            {"Order": "B", "Daily Target": 15, "Good Output Today": 0},
            {"Order": "C", "Daily Target": None, "Good Output Today": 4},
            {"Order": "D", "Daily Target": 0, "Good Output Today": 0},
        ])
        allocation = pd.DataFrame([
            {"Process": "Cutting", "Workers": 3},
            {"Process": "Cutting", "Workers": 2},
            {"Process": "Sewing", "Workers": 1},
            {"Process": "Skiving", "Workers": 0},
        ])
        output, staffing = overview_chart_data(progress, allocation)
        self.assertEqual(output["Order"].tolist(), ["A", "B"])
        self.assertEqual(output["Good Output Today"].tolist(), [5, 0])
        self.assertEqual(dict(zip(staffing["Process"], staffing["Workers"])), {"Cutting": 5, "Sewing": 1})
