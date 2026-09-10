# RAPID: connect permanent shared storage

RAPID is prepared to save factory settings, orders, attendance, machine conditions, daily actuals, plan snapshots, production notes and corrective actions in PostgreSQL. Admin and Planner sessions use the same saved workspace.

**A hosted database still needs to be created in your account and connected through Streamlit Secrets.** This guide does not create a provider account, purchase a plan or activate cloud storage automatically.

## 1. Create one PostgreSQL database for RAPID

Use a managed PostgreSQL provider you can administer. Create a dedicated RAPID project/database, choose a region suitable for your factory and app host, and enable the provider's backup or recovery service. Keep the account recovery details with the person responsible for RAPID.

Two optional providers are:

- **Neon:** create a project, then choose **Connect** in its dashboard. Select the database and role and copy the complete connection string. Keep the provider's TLS/security parameters, including `sslmode=require` and `channel_binding=require` when supplied. See [Neon's connection instructions](https://neon.com/docs/connect/query-with-psql-editor) and [connection security defaults](https://neon.com/docs/changelog/2025-06-27).
- **Supabase:** create a project and use **Connect** to obtain the PostgreSQL connection string. For an app host that cannot reach the direct IPv6 endpoint, choose the **Session pooler** connection, normally port `5432`. Copy its exact username, host and database. RAPID needs the database connection string, not a Supabase project URL or API key. See [Supabase's connection methods](https://supabase.com/docs/guides/database/connecting-to-postgres).

Use the provider's supported secure connection settings. Do not remove TLS parameters to work around a connection failure. If a connection string contains a password placeholder, replace it privately; special password characters must be correctly URL-encoded. The provider-generated complete string is preferable to assembling one manually.

For the database administrator: the RAPID database role needs to connect to its database, use and create tables in its default schema (normally `public`), and read, insert and update RAPID's tables. A role that owns this dedicated schema and its tables can do this. Server-wide superuser access is unnecessary. Keep the role's default schema stable and avoid granting access to unrelated production databases.

## 2. Store the connection and login credentials securely

For the deployed app, open your Streamlit Community Cloud workspace, find RAPID, open its menu, select **Settings**, then **Secrets**. Add the following top-level entries, replacing every placeholder privately, and save. This is the supported [Streamlit app settings workflow](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-settings).

```toml
RAPID_DATABASE_URL = "postgresql://DATABASE_USER:ENCODED_PASSWORD@DATABASE_HOST:5432/DATABASE_NAME?sslmode=require"

ADMIN_USERNAME = "YOUR_ADMIN_USERNAME"
ADMIN_PASSWORD = "YOUR_UNIQUE_ADMIN_PASSWORD"
USER_USERNAME = "YOUR_PLANNER_USERNAME"
USER_PASSWORD = "YOUR_DIFFERENT_UNIQUE_PLANNER_PASSWORD"
```

Use distinct Admin and Planner usernames and strong, different passwords. All four login entries must be explicitly configured for shared PostgreSQL mode. Replace the example database URL with the complete provider URL, preserving any extra security options it contains.

The database password belongs only to the app administrator. Planners sign in with the Planner credentials and do not need the database password. These are two shared application roles; this version does not provide individual employee accounts, password reset emails or single sign-on.

Do not paste real passwords in chat, commit them to GitHub or put them in this guide. For local development, `.streamlit/secrets.toml` is excluded from Git; `.streamlit/secrets.toml.example` is only a placeholder template. Streamlit recommends keeping real secrets outside your repository. See [Streamlit Secrets management](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).

On another host, `RAPID_DATABASE_URL` may instead be provided through the host's secret environment-variable settings. It takes precedence over the same key in Streamlit Secrets. Keep the four login values in Streamlit Secrets. RAPID accepts a normal `postgresql://` URL and selects its PostgreSQL driver internally.

## 3. Start the app and verify the first saved record

After the updated app and secrets are available, restart or reload RAPID and sign in as Admin. On the first successful connection after login, RAPID creates these tables if they do not exist:

| Table | Saved information |
| --- | --- |
| `rapid_documents` | Current factory configuration and operating inputs |
| `rapid_daily_records` | Latest saved plan and inputs for each production date |
| `rapid_replans` | Individual saved replan snapshots with revision, time and username |
| `rapid_configuration_history` | Saved Admin configuration revisions |

Existing saved records are not reset when the app restarts. Do not drop tables or reset the database schema during routine deployment.

1. Confirm that the sidebar says **Shared PostgreSQL database**.
2. In **Admin Settings**, replace example workforce, capacity and machine references with verified factory values. Click **Save Admin Settings**.
3. On the Dashboard, set the correct planning date and review attendance, orders, materials and machine availability. Initial example orders are demonstration data; replace them with the orders you intend to plan.
4. Click **REPLAN TODAY**. Wait for a successful save message and a saved plan revision/time. A forecast changing on screen does not, by itself, confirm a database save.
5. Open a separate browser session and sign in as Planner. Confirm that the saved date, settings and orders are present.
6. Record a small, accurate update, save it, then reload saved data in the first session and confirm both sessions agree. Reopen the app to verify the record remains available.

Start with a pilot line and real, checked figures. Completing these checks confirms your particular database and hosted app connection; successful local tests alone do not establish that the cloud connection works.

## 4. Use the daily workflow

**Edit → review the forecast → save → confirm the saved revision.**

- Dashboard edits are session drafts until saved. **REPLAN TODAY** saves the operating inputs, daily actual production, calculated plan, production notes and corrective actions together. In the **Targets · Losses · Actions** tab, first use **Apply production entries**, **Apply action updates** or **Add action to draft**, then **REPLAN TODAY**. The Apply/Add buttons update the draft; they do not confirm a database save.
- Admin reference changes require **Save Admin Settings** before saving a plan based on those values.
- Other sessions see saved changes after reopening or choosing **Reload saved data (discard draft)**. Reload intentionally replaces unsaved edits. Copy any unsaved information you still need before reloading.
- If another user saves first, RAPID rejects a stale save instead of overwriting the newer record. Preserve your intended changes, reload the saved workspace, review the other user's update, reapply the still-relevant changes and save again.
- If saving fails, the draft remains in that session. Check the connection and retry. Do not close the browser assuming the draft was saved.

### Moving to the next production date

Finish and save the current date before selecting a later date. RAPID carries saved **Actual Production Today** into **Completed Before Today** once, then resets daily output and daily workers used for the new date. Attendance and machine conditions carry forward as editable assumptions: verify them for the new shift.

Use dates in chronological order. RAPID does not invent production for dates you skip. A later-date draft is not a saved daily record until you save it. Earlier saved dates are read-only; choose the latest date to continue planning. If an earlier actual was wrong, preserve the historical record and document a correction on the current date after checking cumulative quantities.

Saved historical forecasts are snapshots of the inputs and factory configuration used at the time. They remain available even when today's settings change. Hypothetical what-if scenarios do not replace the active saved plan.

## 5. Keep recovery copies

Use **Saved daily records & data backup → Download complete data backup** for a JSON export of RAPID's saved data tables. Store the download securely: it contains production information. It excludes the database connection string and login credentials.

**The JSON download is a data export, not an automatic restore feature.** This version has no JSON import/restore button. Enable provider-managed backups and understand the provider's recovery procedure before relying on RAPID for production records. CSV reports are useful for review, but are not a complete database recovery copy.

Connecting a new empty PostgreSQL database does not automatically import an old SQLite database, session history or a JSON export. Plan and validate any data migration separately before switching a live production database.

## 6. Troubleshooting

| What you see | What to check |
| --- | --- |
| Permanent storage is not configured | Add `RAPID_DATABASE_URL` to Streamlit Secrets and reload the app. |
| Login configuration is required | Supply all four explicit Admin/Planner entries above in Streamlit Secrets. |
| Database cannot complete the operation | Check that the provider database is running, the URL/password are correct, the network endpoint is reachable and the required schema permissions exist. Keep the draft open until saving succeeds. |
| PostgreSQL host cannot be reached | Verify the host's network restrictions and provider connection method. For Supabase on an IPv4-only host, use its Session pooler URL. |
| Another user or Admin settings changed | Preserve your intended edits, reload saved data, reconcile changes and replan. |
| An earlier date cannot be edited | Historical dates are read-only. Open the latest date to continue operations. |
| The sidebar says Local database | The app is using SQLite; connect PostgreSQL for shared cloud persistence. |

If you share diagnostic logs, remove database URLs, passwords and production details first. Never resolve a connection issue by placing credentials in source code or deleting RAPID tables.

### Optional: explicit local-only storage

For a local demonstration on one computer, use this value in the private `.streamlit/secrets.toml` file:

```toml
RAPID_DATABASE_URL = "sqlite:///.rapid-data/rapid.sqlite3"
```

Run the app from the project directory so the relative path stays consistent. This stores records in a local file, survives local app restarts and can be shared by sessions using that same app instance. It is not hosted durable storage and does not share data across separate computers or deployment instances. Do not use SQLite for the deployed Streamlit Community Cloud production workspace.
