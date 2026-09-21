"""Charts must describe the computed plan without changing or inventing data."""
import ast
from pathlib import Path
import textwrap
import unittest
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def chart_helpers():
    source = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
    ns = {"pd": pd, "plt": plt, "textwrap": textwrap,
          "st": SimpleNamespace(session_state=SimpleNamespace(planning_date="2026-09-05"))}
    for node in source.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("CHART_") for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), ns)
        if isinstance(node, ast.FunctionDef) and (node.name.startswith("plot_") or node.name in {"_chart_axes", "_add_bar_labels"}):
            node.decorator_list = []
            exec(compile(ast.Module(body=[node], type_ignores=[]), "app.py", "exec"), ns)
    return ns


class ChartDataTests(unittest.TestCase):
    def setUp(self):
        self.h = chart_helpers()
        self.addCleanup(plt.close, "all")

    def test_risk_counts_match_classifier_labels(self):
        df = pd.DataFrame({"ML Delivery Risk": ["LOW", "HIGH", "HIGH", "MEDIUM"]})
        fig = self.h["plot_risk_chart"](df)
        self.assertEqual([p.get_width() for p in fig.axes[0].patches], [2, 1, 1])

    def test_workforce_never_substitutes_future_day_for_selected_date(self):
        daily = pd.DataFrame({"production_date": ["2026-09-07"], "order": ["A"], "workers_allocated": [8]})
        self.assertIsNone(self.h["plot_daily_workforce"](daily, [], "2026-09-05"))
        fig = self.h["plot_daily_workforce"](daily, [], "2026-09-07")
        self.assertTrue(any("Not recorded" in t.get_text() for t in fig.axes[0].get_yticklabels()))

    def test_actuals_use_latest_saved_revision_and_keep_zero(self):
        daily = pd.DataFrame({"production_date": ["2026-09-07"], "order": ["A"], "workers_allocated": [8]})
        history = [{"planning_date": "2026-09-07", "actual_workers": {"A": 6}}, {"planning_date": "2026-09-07", "actual_workers": {"A": 0}}]
        fig = self.h["plot_daily_workforce"](daily, history, "2026-09-07")
        self.assertEqual(float(fig.axes[0].images[0].get_array()[0, 0]), 8)
        self.assertIn("Actual 07 Sep: 0", fig.axes[0].get_yticklabels()[0].get_text())

    def test_changes_compare_each_metric_in_its_own_units(self):
        df = pd.DataFrame({"Indicator": ["Workers Present", "Projected Delay (Days)"], "Previous Plan": [150, 3], "Today": [160, 1], "Change": ["+10", "-2"]})
        fig = self.h["plot_change_chart"](df)
        self.assertEqual(len(fig.axes), 2)
        self.assertEqual([p.get_width() for p in fig.axes[0].patches], [150, 160])
        self.assertEqual([p.get_width() for p in fig.axes[1].patches], [3, 1])

    def test_scenario_totals_and_delay_match_table(self):
        df = pd.DataFrame({"Scenario": ["Current Operating Plan", "Disruption"], "Orders On Time": ["3/3", "1/3"], "Projected Delay (Days)": [0, 5]})
        before = df.copy(deep=True)
        fig = self.h["plot_scenario_chart"](df)
        self.assertEqual([p.get_width() for p in fig.axes[0].patches], [3, 1, 0, 2])
        self.assertEqual([p.get_width() for p in fig.axes[1].patches], [0, 5])
        pd.testing.assert_frame_equal(df, before)

    def test_unfinished_resource_is_not_coloured_on_time(self):
        df = pd.DataFrame({"Order": ["A"], "Recommended Workers Today": [0], "Projected Delay (Days)": [0], "On Time?": ["NO"]})
        fig = self.h["plot_resource_chart"](df)
        self.assertIn("unfinished", fig.axes[0].get_yticklabels()[0].get_text())


if __name__ == "__main__":
    unittest.main()
