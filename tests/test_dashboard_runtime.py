"""Real Streamlit smoke checks; install requirements.txt before running."""
import unittest
import os
import tempfile
from unittest.mock import patch
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:
    AppTest = None


@unittest.skipIf(AppTest is None, "Install requirements.txt for Streamlit runtime tests")
class DashboardRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.env = patch.dict(os.environ, {"RAPID_DATABASE_URL": "sqlite:///" + (Path(self.temp.name) / "rapid.sqlite3").as_posix()})
        self.env.start()
        self.addCleanup(self.env.stop)
        # Cached engines must release Windows database handles before temp cleanup.
        if AppTest is not None:
            import streamlit as st
            self.addCleanup(st.cache_resource.clear)

    def login(self, role="admin"):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60).run()
        self.assertEqual(len(app.exception), 0)
        app.text_input[0].set_value(role)
        app.text_input[1].set_value("admin2026" if role == "admin" else "user2026")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        return app

    def click(self, app, label):
        next(b for b in app.button if label in b.label).click().run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def test_replanning_attendance_and_all_change_indicators(self):
        app = self.login()
        self.click(app, "REPLAN TODAY")
        next(n for n in app.number_input if n.label == "Workers Present Today").set_value(110).run()
        self.click(app, "REPLAN TODAY")
        self.assertEqual(app.session_state.plan_history[-1]["workers_present"], 110)
        self.assertEqual(sum(app.session_state.plan_history[-1]["recommended_workers"].values()), 110)
        changes = next(d.value for d in app.dataframe if "Indicator" in d.value.columns)
        self.assertEqual(len(changes), 7)

    def test_all_recovery_actions_keep_active_orders_unchanged(self):
        app = self.login()
        original = app.session_state.orders.copy(deep=True)
        for action in next(s for s in app.selectbox if s.label == "Recovery action to test").options:
            next(s for s in app.selectbox if s.label == "Recovery action to test").select(action).run()
            self.click(app, "Test Recovery Action")
            self.assertTrue(app.session_state.orders.equals(original))
            self.assertEqual(len(app.metric), 4)

    def test_admin_and_reports_pages(self):
        app = self.login()
        app.radio[0].set_value(next(v for v in app.radio[0].options if "Reports" in v)).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        labels = {button.label for button in app.get("download_button")}
        self.assertTrue({"Order Results CSV", "Daily Allocation CSV", "Machine Status CSV", "Decision Centre CSV",
                         "Material Tracker CSV", "Replan History CSV", "Management Summary TXT"}.issubset(labels))
        app.radio[0].set_value(next(v for v in app.radio[0].options if "Admin Settings" in v)).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        next(n for n in app.number_input if n.label == "Standard Production Capacity (uppers/day)").set_value(320.0).run()
        self.assertEqual(app.session_state.settings["base_capacity"], 320)
        self.click(app, "Save Admin Settings")
        reopened = self.login("planner")
        self.assertEqual(reopened.session_state.settings["base_capacity"], 320)

    def test_saved_plan_survives_signout_and_a_separate_planner_session(self):
        app = self.login()
        next(n for n in app.number_input if n.label == "Workers Present Today").set_value(117).run()
        self.click(app, "REPLAN TODAY")
        self.click(app, "SIGN OUT")
        reopened = self.login("planner")
        self.assertEqual(reopened.session_state.workers_present, 117)
        self.assertEqual(reopened.session_state.plan_history[-1]["workers_present"], 117)

    def test_stale_planner_does_not_overwrite_saved_attendance(self):
        first = self.login("planner")
        second = self.login("planner")
        next(n for n in first.number_input if n.label == "Workers Present Today").set_value(115).run()
        self.click(first, "REPLAN TODAY")
        next(n for n in second.number_input if n.label == "Workers Present Today").set_value(90).run()
        self.click(second, "REPLAN TODAY")
        self.assertTrue(any("Another user" in error.value for error in second.error))
        reopened = self.login("planner")
        self.assertEqual(reopened.session_state.workers_present, 115)

    def test_planner_permissions_zero_attendance_and_reports(self):
        app = self.login("planner")
        self.assertFalse(any("Admin Settings" in v for v in app.radio[0].options))
        next(n for n in app.number_input if n.label == "Workers Present Today").set_value(0).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        app.radio[0].set_value(next(v for v in app.radio[0].options if "Reports" in v)).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        results = next(d.value for d in app.dataframe if "completion_day" in d.value.columns)
        self.assertTrue(results.completion_day.isna().all())
        self.assertFalse((results.on_time == "YES").any())


if __name__ == "__main__":
    unittest.main()
