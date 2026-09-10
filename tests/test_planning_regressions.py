"""Calculation regressions; run with python -m unittest discover -s tests -v.

App helpers are extracted without starting Streamlit or fitting the ML model.
Their production function bodies execute unchanged with explicit test inputs.
"""
import ast
import pathlib
import types
import unittest

import pandas as pd
import planner_core as core


ROOT = pathlib.Path(__file__).resolve().parents[1]


def app_helpers():
    tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
    ns = {"pd": pd, "business_days_until_ready": core.business_days_until_ready}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "BLOCKED_MATERIAL_STATUSES"
            for t in node.targets
        ):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and node.name == "material_delay_info":
            exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), ns)
    return ns


def order(name="A", quantity=100, due=2, machine=1, delay=0):
    return dict(order=name, quantity=quantity, due_day=due,
                machine_availability=machine, material_delay_days=delay, risk="LOW")


def simulate(orders, labour=1, horizon=120):
    return core.simulate_orders(orders, 300, 150, labour, 0, horizon=horizon)


class PlanningRegressions(unittest.TestCase):
    def test_unfinished_has_no_completion_and_is_never_on_time(self):
        for orders, labour in [([order(due=200)], 0),
                               ([order(delay=120, due=200)], 1),
                               ([order(quantity=100000, due=200)], 1)]:
            with self.subTest(orders=orders, labour=labour):
                results, daily, _ = simulate(orders, labour)
                self.assertTrue(pd.isna(results.iloc[0].completion_day))
                self.assertNotEqual(results.iloc[0].on_time, "YES")
                self.assertGreater(daily.iloc[-1].remaining, 0)

    def test_completed_order_still_has_real_completion(self):
        results, _, _ = simulate([order()])
        self.assertEqual(results.iloc[0].completion_day, 1)
        self.assertEqual(results.iloc[0].on_time, "YES")

    def test_stopped_process_does_not_starve_ready_order(self):
        _, daily, _ = simulate([order("stopped", due=1, machine=0), order("ready")], horizon=3)
        self.assertEqual(daily[daily.order == "stopped"].workers_allocated.sum(), 0)
        self.assertGreater(daily[(daily.order == "ready") & (daily.day == 1)].produced.sum(), 0)
        self.assertLessEqual(daily.groupby("day").workers_allocated.sum().max(), 150)

    def test_all_stopped_processes_allocate_zero_workers(self):
        _, daily, _ = simulate([order(machine=0)], horizon=2)
        self.assertEqual(daily.workers_allocated.sum(), 0)

    def test_duplicate_ids_are_rejected_before_simulation(self):
        for ids in [("A", "A"), ("A", " A ")]:
            with self.subTest(ids=ids), self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
                simulate([order(ids[0]), order(ids[1])])

    def test_duplicate_ids_are_rejected_by_allocator(self):
        states = [{**order(), "remaining": 100}, {**order(), "remaining": 100}]
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            core.allocate_workers_for_day(states, 150, 300, 150, 0, 0)

    def test_unique_orders_preserve_attendance(self):
        _, daily, _ = simulate([order("A", quantity=1000), order("B", quantity=1000)], labour=110/150)
        self.assertEqual(daily[daily.day == 1].workers_allocated.sum(), 110)
        self.assertLessEqual(daily.groupby("day").workers_allocated.sum().max(), 110)

    def test_material_scenario_uses_stated_duration(self):
        helper = app_helpers()["material_delay_info"]
        row = {"Material Status": "Ready", "Expected Material Ready Date": pd.NaT}
        for duration in [1, 2, 3, 4]:
            with self.subTest(duration=duration):
                _, delay, _, eligible = helper(row, "2026-09-01", "Quality Hold", duration)
                self.assertEqual(delay, duration)
                self.assertFalse(eligible)
                _, daily, _ = simulate([order(delay=delay)], horizon=duration+1)
                self.assertEqual(daily[daily.day <= duration].produced.sum(), 0)
                self.assertGreater(daily[daily.day == duration+1].produced.sum(), 0)

    def test_real_unknown_material_hold_remains_blocked(self):
        helper = app_helpers()["material_delay_info"]
        row = {"Material Status": "Quality Hold", "Expected Material Ready Date": pd.NaT}
        self.assertEqual(helper(row, "2026-09-01")[1], 120)

    def test_known_material_delay_and_scenario_extension(self):
        helper = app_helpers()["material_delay_info"]
        row = {"Material Status": "Awaiting Material", "Expected Material Ready Date": pd.Timestamp("2026-09-04")}
        self.assertEqual(helper(row, "2026-09-01")[1], 3)
        self.assertEqual(helper(row, "2026-09-01", "Quality Hold", 2)[1], 5)

    def test_overdue_working_day_delay(self):
        # Signed scheduling deadline must preserve elapsed lateness, independently of ML's minimum-one-day input.
        deadline = core.business_day_deadline("2026-09-10", "2026-09-01")
        result, _, _ = simulate([order(quantity=10, due=deadline)])
        self.assertEqual(result.iloc[0].projected_delay_days, 7)
        self.assertEqual(result.iloc[0].on_time, "NO")
        self.assertEqual(core.business_days_between("2026-09-10", "2026-09-01"), 1)

    def test_due_today_and_future_deadlines(self):
        for due, expected in [("2026-09-10", 1), ("2026-09-11", 2), ("2026-09-13", 2)]:
            with self.subTest(due=due):
                self.assertEqual(core.business_day_deadline("2026-09-10", due), expected)

    def test_weekend_schedule_starts_monday(self):
        for start in ["2026-09-12", "2026-09-13"]:
            with self.subTest(start=start):
                dates = [str(core.add_business_days(start, i).date()) for i in range(3)]
                self.assertEqual(dates, ["2026-09-14", "2026-09-15", "2026-09-16"])

    def test_weekday_schedule_and_material_release_stay_aligned(self):
        self.assertEqual(str(core.add_business_days("2026-09-11", 1).date()), "2026-09-14")
        self.assertEqual(core.business_days_until_ready("2026-09-12", "2026-09-14"), 0)

    def test_weekend_past_deadline_is_late_on_monday(self):
        deadline = core.business_day_deadline("2026-09-12", "2026-09-13")
        result, _, _ = simulate([order(quantity=10, due=deadline)])
        self.assertEqual(result.iloc[0].projected_delay_days, 1)

    def test_completion_display_and_export_allow_missing_dates(self):
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        checked = 0
        ns = {"pd": pd, "add_business_days": core.add_business_days,
              "st": types.SimpleNamespace(session_state=types.SimpleNamespace(planning_date="2026-09-01"))}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and "completion_day" in ast.unparse(node) and isinstance(node.value, ast.Call):
                for child in ast.walk(node.value):
                    if isinstance(child, ast.Lambda):
                        formatter = eval(compile(ast.Expression(child), "app.py", "eval"), ns)
                        for missing in [None, float("nan"), pd.NA]:
                            value = formatter(missing)
                            self.assertTrue(pd.isna(value) or value in ("—", "Not scheduled"))
                        checked += 1
        self.assertEqual(checked, 2)


class AppCalculationIntegration(unittest.TestCase):
    """Run the real calculation wiring with only Streamlit and ML stubbed."""

    def setUp(self):
        self.ns = app_helpers()
        self.errors = []
        self.features = []
        self.session = types.SimpleNamespace(planning_date="2026-09-10", workers_present=150)

        def stop():
            raise RuntimeError("validation stopped calculation")

        def predict(features):
            self.features.append(features)
            return "LOW", 80.0

        self.ns.update({
            "st": types.SimpleNamespace(session_state=self.session, error=self.errors.append, stop=stop),
            "settings": dict(benchmark_workers=150, base_capacity=300, max_overtime=0.05),
            "normalize_orders": lambda frame: frame.copy(),
            "process_options": lambda: ["P"],
            "machine_availability_map": lambda: ({"P": 1.0}, 1.0, "P", 1, 1, 1.0),
            "predict_risk": predict,
            "risk_drivers": lambda *args: "test operational indicator",
        })
        for name in ["business_days_between", "business_day_deadline", "validate_order_ids",
                     "recommend_overtime", "simulate_orders", "build_recommendations", "add_business_days"]:
            self.ns[name] = getattr(core, name)
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in {"calculate_plan", "management_decision_rows"}:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), self.ns)
        self.orders = pd.DataFrame([{
            "Order": "A", "Original Quantity": 10, "Completed Before Today": 0,
            "Actual Production Today": 0, "Due Date": pd.Timestamp("2026-09-01"),
            "Material Status": "Ready", "Expected Material Ready Date": pd.NaT,
            "Current Process": "P",
        }])

    def test_overdue_deadline_is_used_without_changing_ml_days(self):
        bundle = self.ns["calculate_plan"](orders_source=self.orders)
        self.assertEqual(bundle[3].iloc[0].projected_delay_days, 7)
        self.assertEqual(bundle[3].iloc[0].on_time, "NO")
        self.assertEqual(self.features[0]["days_remaining"], 1)

    def test_unfinished_plan_recommendations_and_decisions(self):
        for due, expected_status in [("2026-09-01", "NO"), ("2027-09-01", "UNKNOWN")]:
            with self.subTest(due=due):
                self.orders["Due Date"] = pd.Timestamp(due)
                bundle = self.ns["calculate_plan"](orders_source=self.orders, workers_override=0)
                result = bundle[3].iloc[0]
                self.assertTrue(pd.isna(result.completion_day))
                self.assertEqual(result.on_time, expected_status)
                self.assertTrue(result.projected_delay_is_lower_bound)
                self.assertEqual(bundle[6].iloc[0]["status"], "NOT COMPLETED WITHIN HORIZON")
                decisions = self.ns["management_decision_rows"](bundle[3], bundle[0], 0, pd.DataFrame())
                self.assertIn("No completion within planning horizon", decisions.Issue.tolist())

    def test_duplicate_app_input_stops_before_prediction(self):
        duplicates = pd.concat([self.orders, self.orders], ignore_index=True)
        with self.assertRaisesRegex(RuntimeError, "validation stopped"):
            self.ns["calculate_plan"](orders_source=duplicates)
        self.assertIn("Duplicate order IDs", self.errors[0])
        self.assertEqual(self.features, [])

    def test_material_scenario_and_recovery_use_real_duration(self):
        self.orders["Due Date"] = pd.Timestamp("2026-09-30")
        for delay in [3, 1]:
            with self.subTest(delay=delay):
                bundle = self.ns["calculate_plan"](orders_source=self.orders,
                    extra_delay_map={"A": delay}, status_override_map={"A": "Quality Hold"})
                self.assertEqual(bundle[0][0]["material_delay_days"], delay)
                self.assertEqual(bundle[3].iloc[0].completion_day, delay + 1)
        self.assertEqual(self.orders.iloc[0]["Material Status"], "Ready")

    def test_weekend_dates_flow_into_daily_schedule(self):
        self.session.planning_date = "2026-09-12"
        self.orders["Original Quantity"] = 700
        self.orders["Due Date"] = pd.Timestamp("2026-09-30")
        daily = self.ns["calculate_plan"](orders_source=self.orders)[4]
        self.assertEqual([str(d) for d in daily.production_date],
                         ["2026-09-14", "2026-09-15", "2026-09-16"])


if __name__ == "__main__":
    unittest.main()
