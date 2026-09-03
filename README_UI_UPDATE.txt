RAPID v3.1 — MANAGEMENT UI REFRESH
=================================

RAPID = Resource Allocation & Production Intelligence for Delivery

This package keeps the v3 planning/ML logic and rebuilds the Streamlit presentation to more closely match the approved management-dashboard reference.

KEY UI CHANGES
- Renamed the application from SURE-Plan to RAPID.
- Management-first dashboard: compact top controls, KPI cards, risk/allocation panels, machine/material visibility, workforce chart, decision centre and what-if analysis.
- Large daily order and machine editors are moved into a collapsed "Update today's orders, production, materials & machine status" panel.
- Added a forced light Streamlit theme through .streamlit/config.toml so data editors/tables no longer inherit a dark browser theme.
- Preserved process-wise machine breakdown visibility, material readiness, rolling replanning, daily change monitor, Recommended Workers Today, Management Decision Centre and What-if Impact Analysis.

GITHUB UPDATE
1. Replace app.py in the repo root.
2. Keep/replace planner_core.py with the included version (logic is unchanged from v3 final).
3. Upload .streamlit/config.toml into the repo's .streamlit folder. This is important for the light dashboard/table appearance.
4. Keep prototype_training_data.csv and requirements.txt in the repo root.
5. Commit and push to main.
6. Reboot the Streamlit app if the new theme does not appear immediately.
