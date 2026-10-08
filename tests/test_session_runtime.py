"""No-database deployment must open, replan and isolate users."""
import os
from pathlib import Path
import unittest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest


class SessionRuntimeTests(unittest.TestCase):
    def login(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=60)
        app.secrets["RAPID_DATABASE_URL"] = ""
        app.run()
        app.text_input[0].set_value("admin")
        app.text_input[1].set_value("admin2026")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    @patch.dict(os.environ, {"RAPID_DATABASE_URL": ""})
    def test_without_database_can_save_and_other_session_is_isolated(self):
        with patch("rapid_storage.create_engine", side_effect=AssertionError("No database allowed")):
            app = self.login()
            self.assertTrue(any("Session-only" in c.value for c in app.caption))
            next(n for n in app.number_input if n.label == "Workers Present Today").set_value(113).run()
            next(b for b in app.button if "Save today's plan" in b.label).click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.session_state.plan_history[-1]["workers_present"], 113)
            next(b for b in app.button if "Reload saved data" in b.label).click().run()
            self.assertEqual(app.session_state.workers_present, 113)
            other = self.login()
            self.assertEqual(other.session_state.workers_present, 150)
            self.assertEqual(other.session_state.plan_history, [])

    @patch.dict(os.environ, {"RAPID_DATABASE_URL": ""})
    def test_simple_daily_entry_keeps_accepted_target_fixed(self):
        app = self.login()
        overview = next(d.value for d in app.dataframe if "Recommended workers" in d.value.columns)
        planned_row = overview.loc[overview["Daily Target"] > 0].iloc[0]
        order = planned_row["Order"]
        planned = int(planned_row["Daily Target"])
        self.assertGreater(planned, 0)
        next(b for b in app.button if b.label == "Save today's plan").click().run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        accepted = next(n for n in app.session_state.production_notes if n["order"] == order)
        self.assertEqual(accepted["daily_target"], planned)
        next(s for s in app.selectbox if s.label == "Order to update").select(order).run()
        next(n for n in app.number_input if n.label == "Good units completed today").set_value(10).run()
        next(b for b in app.button if b.label == "Update actual output").click().run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])
        self.assertEqual(int(app.session_state.orders.loc[app.session_state.orders["Order"] == order, "Actual Production Today"].iloc[0]), 10)
        progress = next(d.value for d in app.dataframe if "Recommended workers" in d.value.columns)
        row = progress.loc[progress["Order"] == order].iloc[0]
        self.assertEqual(int(row["Daily Target"]), planned)
        self.assertEqual(int(row["Target Remaining"]), max(0, planned - 10))


if __name__ == "__main__":
    unittest.main()
