"""Real Streamlit form checks for production entries and action ownership."""
import unittest

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError:
    AppTest = None


APP_SOURCE = '''
from datetime import date
import pandas as pd
import streamlit as st
from production_management import render_production_management

if "orders" not in st.session_state:
    st.session_state.orders = pd.DataFrame([
        {"Order": "A", "Current Process": "Stitching", "Original Quantity": 1000,
         "Completed Before Today": 250, "Actual Production Today": 210},
        {"Order": "B", "Current Process": "Cutting", "Original Quantity": 800,
         "Completed Before Today": 100, "Actual Production Today": 100},
    ])
    st.session_state.planning_date = date(2026, 9, 10)
    st.session_state.username = "Supervisor"
    st.session_state.production_notes = []
    st.session_state.actions = []
if render_production_management(st.session_state, read_only=st.session_state.get("readonly", False), key_suffix="test"):
    st.session_state.draft_changed = True
    st.rerun()
'''


@unittest.skipIf(AppTest is None, "Install requirements.txt for Streamlit UI checks")
class ProductionUITests(unittest.TestCase):
    def app(self):
        app = AppTest.from_string(APP_SOURCE).run()
        self.no_errors(app)
        return app

    def no_errors(self, app):
        self.assertEqual(len(app.exception), 0, [e.message for e in app.exception])

    def click(self, app, label):
        next(b for b in app.button if b.label == label).click().run()
        self.no_errors(app)

    def test_daily_target_form_preserves_authoritative_actuals(self):
        app = self.app()
        original = app.session_state.orders.copy(deep=True)
        key = "production_notes_editor_test_2026-09-10"
        app.session_state[key] = {
            "edited_rows": {0: {"Daily Target": 300, "Downtime Minutes": 45,
                                "Recorded Reason": "Machine repair ticket 42"}},
            "added_rows": [], "deleted_rows": [],
        }
        self.click(app, "Apply production entries")
        self.assertEqual(app.session_state.production_notes[0]["daily_target"], 300)
        self.assertEqual(app.session_state.production_notes[0]["process"], "Stitching")
        self.assertEqual(app.session_state.production_notes[0]["downtime_minutes"], 45)
        self.assertTrue(app.session_state.orders.equals(original))
        progress = next(d.value for d in app.dataframe if "Target Remaining" in d.value.columns)
        self.assertEqual(progress.loc[0, "Target Remaining"], 90)

    def test_action_form_and_status_update_preserve_orders(self):
        app = self.app()
        original = app.session_state.orders.copy(deep=True)
        app.selectbox[0].select("A")
        next(t for t in app.text_input if t.label == "Action to take").set_value("Inspect stitching machine")
        next(t for t in app.text_input if t.label == "Responsible person").set_value("Maintenance supervisor")
        app.text_area[0].set_value("Machine stopped at 10:00")
        self.click(app, "Add action to draft")
        self.assertEqual(len(app.session_state.actions), 1)
        self.assertEqual(app.session_state.actions[0]["process"], "Stitching")
        self.assertEqual(app.session_state.actions[0]["owner"], "Maintenance supervisor")
        self.assertEqual(app.session_state.actions[0]["status"], "Open")
        self.assertTrue(app.session_state.orders.equals(original))
        app.session_state["actions_editor_test_2026-09-10"] = {
            "edited_rows": {0: {"Status": "Done"}}, "added_rows": [], "deleted_rows": [],
        }
        self.click(app, "Apply action updates")
        self.assertEqual(app.session_state.actions[0]["status"], "Done")
        self.assertTrue(app.session_state.orders.equals(original))

    def test_incomplete_action_is_rejected_without_creating_record(self):
        app = self.app()
        next(t for t in app.text_input if t.label == "Action to take").set_value("Inspect stitching machine")
        self.click(app, "Add action to draft")
        self.assertEqual(app.session_state.actions, [])
        self.assertTrue(any("owner" in e.value for e in app.error))

    def test_read_only_has_no_edit_forms(self):
        app = self.app()
        app.session_state.readonly = True
        app.run()
        self.no_errors(app)
        self.assertEqual(len(app.button), 0)
        self.assertEqual(len(app.text_input), 0)


if __name__ == "__main__":
    unittest.main()
