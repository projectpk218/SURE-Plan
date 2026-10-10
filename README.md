# RAPID — Production Planning and Daily Operations Dashboard

### Daily planner workflow

1. **Overview** explains the current production position and links to each workflow page.
2. **1 · Setup inputs**: confirm the production date and attendance. Add or update an order with the short form; update a machine process with its availability form. The full order grid, material fields and machine grid remain available for bulk edits. The sample workspace starts on 1 Sep 2026; select the real production date for live use.
3. **2 · Worker plan**: review workers, process and expected output for each order. Workers and output come from the same scheduled date. Choose an order to read the recommendation and its supporting reasons. **Confirm these worker assignments** records actual staffing only after the planner confirms deployment on the floor; it does not invent production output. Production runs Monday–Sunday: Saturdays and Sundays receive targets and staffing just like weekdays. Material waiting periods, due-date calculations and delivery delays count all calendar days. Previously saved daily targets remain fixed; adjust an existing saved zero target explicitly if needed.
4. **3 · Production progress**: enter each order's total good units made today and optional actual staffing. RAPID calculates target left, target reached and total order balance. Daily targets come from the day-start plan and remain fixed once saved. Their purpose differs from the latest remaining-quantity forecast.
5. **4 · Delivery & actions**: compare due dates with forecast completion and delay. Record reasons, adjust a target, or assign corrective actions with owners and follow-up dates. Unfinished orders show no invented completion date; delay lower bounds remain labelled.
6. **Detailed analysis** retains ML evidence, process allocation, machines, materials, change history, scenarios and the complete rolling schedule. **Reports** and **Admin Settings** remain separate sidebar pages.

Use **Save today's plan** on any workflow page to save the current draft. Draft values survive page navigation. Save or discard before changing production dates. Expected output is a full working-day model estimate based on current remaining order quantities; it does not account for hours remaining in a partly completed shift. Actual worker entries are records and do not override the calculation engine's automatic allocation.

Actual output reduces the remaining order quantity. A daily target is a tracking commitment; it does not directly change the planner's capacity or delivery calculation.

RAPID is a rolling daily production planning and decision-support tool for concurrent make-to-order shoe-upper production. It combines explainable planning calculations with a shared daily operating record so planners and managers can work from the same saved plan.

### Run without a database

No secrets or database are required for session-only mode. Leave `RAPID_DATABASE_URL` unset and launch `app.py`. Demo logins are Admin `admin` / `admin2026` and Planner `planner` / `user2026`; optional login secrets can replace these. Planning, daily entries, actions and exports work. Saved plans and settings belong only to the current browser session and can be lost after refresh, sign-out or restart. Download reports before leaving. Separate users do not share session records.

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

Follow [STORAGE_SETUP.md](STORAGE_SETUP.md) if permanent shared records are wanted later. Without a database URL, RAPID runs in temporary session-only mode and labels that limitation in the sidebar.

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
