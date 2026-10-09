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
        self.navigate(app, "Setup inputs")
        return app

    def navigate(self, app, page):
        app.radio[0].set_value(next(v for v in app.radio[0].options if page in v)).run()
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

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
        self.navigate(app, "Production progress")
        overview = next(d.value for d in app.dataframe if "Target Remaining" in d.value.columns)
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
        self.navigate(app, "Worker plan")
        self.navigate(app, "Production progress")
        progress = next(d.value for d in app.dataframe if "Target Remaining" in d.value.columns)
        row = progress.loc[progress["Order"] == order].iloc[0]
        self.assertEqual(int(row["Daily Target"]), planned)
        self.assertEqual(int(row["Target Remaining"]), max(0, planned - 10))

    @patch.dict(os.environ, {"RAPID_DATABASE_URL": ""})
    def test_confirm_assignments_survive_navigation_without_inventing_output(self):
        app = self.login()
        next(n for n in app.number_input if n.label == "Workers Present Today").set_value(111).run()
        next(b for b in app.button if b.label.startswith("Next: Worker plan")).click().run()
        self.assertEqual(app.session_state.workers_present, 111)
        self.assertEqual(len(app.exception), 0)
        expected = next(d.value for d in app.dataframe if "Expected output" in d.value.columns)
        next(b for b in app.button if b.label == "Confirm these worker assignments").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(int(app.session_state.orders["Workers Used Today"].sum()), int(expected["Workers"].sum()))
        self.assertEqual(int(app.session_state.orders["Actual Production Today"].sum()), 0)
        self.navigate(app, "Production progress")
        next(n for n in app.number_input if n.label == "Workers actually assigned (optional)").set_value(151)
        next(b for b in app.button if b.label == "Update actual output").click().run()
        self.assertTrue(any("attendance" in e.value for e in app.error))
        self.navigate(app, "Setup inputs")
        self.navigate(app, "Worker plan")
        self.assertEqual(int(app.session_state.orders["Workers Used Today"].sum()), int(expected["Workers"].sum()))

    @patch.dict(os.environ, {"RAPID_DATABASE_URL": ""})
    def test_simple_order_form_reaches_plan_and_preserves_saved_inputs(self):
        app = self.login()
        next(t for t in app.text_input if t.label == "Order ID").set_value("NEW-1")
        next(n for n in app.number_input if n.label == "Total order quantity").set_value(400)
        next(b for b in app.button if b.label == "Apply order details").click().run()
        self.assertEqual(len(app.exception), 0)
        self.navigate(app, "Worker plan")
        view = next(d.value for d in app.dataframe if "Expected output" in d.value.columns)
        self.assertIn("NEW-1", view["Order"].tolist())
        next(b for b in app.button if b.label == "Save today's plan").click().run()
        self.assertEqual(len(app.exception), 0)
        self.navigate(app, "Setup inputs")
        row = app.session_state.orders.loc[app.session_state.orders["Order"] == "NEW-1"].iloc[0]
        self.assertEqual(row["Original Quantity"], 400)


if __name__ == "__main__":
    unittest.main()
