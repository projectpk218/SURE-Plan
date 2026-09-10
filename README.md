# RAPID — Production Planning and Daily Operations Dashboard

RAPID is a rolling daily production planning and decision-support tool for concurrent make-to-order shoe-upper production. It combines explainable planning calculations with a shared daily operating record so planners and managers can work from the same saved plan.

## Main capabilities

- Professional single-column Ocean Blue and Warm Sand dashboard with responsive full-width tables and charts.
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
- Daily targets, good output, downtime/loss observations, production reasons and corrective actions with owners, due dates and follow-up status.
- Persistent configuration, daily records, replan history and JSON backup exports through SQLite locally or PostgreSQL for shared deployment.
- CSV/text report exports and academic/prototype disclosure.

## Important academic disclosure

The ML classifier is a proof-of-concept trained on researcher-designed simulated scenarios. Default workforce, machine-count, and capacity values are prototype/reference assumptions unless they are independently verified using company records. Prototype Model Confidence is a model class probability, not certainty of the actual delivery outcome.

## Run locally

```bash
pip install -r requirements.txt
```

For a local demonstration, create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example` and set a private SQLite URL. Then run:

```bash
streamlit run app.py
```

The local SQLite file survives app restarts on the same computer. It is intended for local use and does not provide shared cloud persistence.

## Deploy with shared records

Use PostgreSQL for a deployed factory workspace. Create the database, add the database URL and explicit Admin/Planner credentials to the deployment's private Streamlit Secrets, then deploy the repository. RAPID creates its tables on first connection and uses optimistic revision checks so one user's save cannot silently overwrite another user's newer save.

Follow [STORAGE_SETUP.md](STORAGE_SETUP.md) for the provider-neutral setup, required secret names, first-save verification, daily workflow, conflict handling and backup guidance. The app intentionally stops with a clear setup message when a deployed instance has no database URL; it does not pretend that session memory is a reliable production record.

For Streamlit Community Cloud, configure secrets in the app's Settings rather than committing them to GitHub. Keep `.streamlit/secrets.toml` private; only the placeholder example belongs in the repository.

## Update an existing GitHub / Streamlit deployment

1. Replace `app.py` in the GitHub repository with the new `app.py`.
2. Replace `planner_core.py` with the new `planner_core.py`.
3. Keep `prototype_training_data.csv` in the repository root (replace it with the packaged copy if needed).
4. Replace/update `requirements.txt` with the packaged version.
5. Configure PostgreSQL and the required private secrets before directing production users to the new release.
6. Commit the changes to the `main` branch and push origin.
7. Streamlit Community Cloud should redeploy automatically. If not, open the app settings and reboot/redeploy the app.

When no shared database is configured locally, RAPID keeps the demo login fallback for convenience. A PostgreSQL deployment requires all four explicit login secrets, so the demo credentials are not silently used in production.
