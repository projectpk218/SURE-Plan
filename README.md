# RAPID v3.0 — Final Consolidated Management Dashboard

RAPID is a rolling daily production planning and decision-support prototype for concurrent make-to-order shoe-upper production.

## Main v3.0 capabilities

- Professional single-page management dashboard with the existing navy + blue/violet/magenta theme.
- Admin/Planner role separation.
- Admin-editable standard workforce (default prototype reference: 150 workers).
- Admin-editable standard production capacity and maximum overtime.
- Process-wise machine master with an editable default reference of 105 machines.
- Daily machine availability, breakdown notes, availability %, and bottleneck identification by production process.
- Material Readiness statuses with expected material-ready dates and allocation blocking when material is unavailable.
- Rolling daily replanning using original quantity, completed production, actual production today, and remaining quantity.
- Daily Change Monitor comparing the previous saved replan with today's conditions.
- Explainable ML delivery-risk assessment with key drivers and Prototype Model Confidence wording.
- Recommended Workers Today plus a process-wise allocation view based on the order's current production process.
- Daily Workforce Allocation by Order using production dates and Actual vs Planned display when actual worker history is entered.
- Management Decision Centre with ACT NOW / ACTION TODAY / MONITOR / NO ACTION priorities.
- What-if Impact Analysis with explicit assumptions, scenario-impact badges, affected orders, reasons, responses, and recovery-action testing.
- CSV/text report exports and academic/prototype disclosure.

## Important academic disclosure

The ML classifier is a proof-of-concept trained on researcher-designed simulated scenarios. Default workforce, machine-count, and capacity values are prototype/reference assumptions unless they are independently verified using company records. Prototype Model Confidence is a model class probability, not certainty of the actual delivery outcome.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Update the existing GitHub / Streamlit deployment

1. Replace `app.py` in the GitHub repository with the new `app.py`.
2. Replace `planner_core.py` with the new `planner_core.py`.
3. Keep `prototype_training_data.csv` in the repository root (replace it with the packaged copy if needed).
4. Replace/update `requirements.txt` with the packaged version.
5. Commit the changes to the `main` branch and Push origin.
6. Streamlit Community Cloud should redeploy automatically. If not, open the app settings and reboot/redeploy the app.

Do **not** upload a real `.streamlit/secrets.toml` containing passwords to a public GitHub repository. Use Streamlit Secrets instead.

Fallback prototype credentials (only when Secrets are not configured):
- Admin: `admin` / `admin2026`
- Planner: `planner` / `user2026`

## Prototype persistence note

Rolling plan history is stored in Streamlit session state in this version. It can reset when a browser session or Streamlit Cloud instance restarts. For permanent multi-day production history, connect the prototype to a persistent database or external data store in a later deployment stage.
