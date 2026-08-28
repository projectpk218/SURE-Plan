from pathlib import Path
from datetime import date
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

from planner_core import (
    business_days_between,
    add_business_days,
    recommend_overtime,
    simulate_orders,
    build_recommendations,
)

BASE = Path(__file__).resolve().parent
TRAINING = pd.read_csv(BASE / "prototype_training_data.csv")
FEATURES = [
    "quantity", "days_remaining", "machine_availability", "labour_availability",
    "material_delay_days", "base_capacity", "required_daily_output", "capacity_ratio"
]

st.set_page_config(
    page_title="SURE-Plan | Production Decision Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Professional management dashboard styling ----------
st.markdown(
    """
    <style>
    :root {
        --navy:#0b2345;
        --navy2:#102f5f;
        --blue:#1769e0;
        --sky:#eef5ff;
        --ink:#172033;
        --muted:#667085;
        --line:#dbe4ef;
        --panel:#ffffff;
        --bg:#f6f8fb;
        --good:#16845b;
        --warn:#b76a00;
        --bad:#c93545;
    }
    .stApp {background: var(--bg); color: var(--ink);}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #0b2345 0%, #102f5f 100%);}
    [data-testid="stSidebar"] * {color:#f8fbff;}
    [data-testid="stSidebar"] .stRadio label {padding:.35rem .55rem; border-radius:8px;}
    [data-testid="stSidebar"] .stRadio label:hover {background:rgba(255,255,255,.08);}
    .block-container {padding-top:1.15rem; padding-bottom:2rem; max-width:1550px;}
    h1,h2,h3 {letter-spacing:-.02em;}
    .sp-brand {display:flex; align-items:center; gap:12px; margin-bottom:18px;}
    .sp-brand-icon {width:42px;height:42px;border-radius:10px;background:#1769e0;display:flex;align-items:center;justify-content:center;font-size:22px;}
    .sp-brand-title {font-size:25px;font-weight:800;line-height:1;color:white;}
    .sp-brand-sub {font-size:12px;opacity:.78;margin-top:4px;}
    .top-head {background:linear-gradient(110deg,#0b2345,#163d73);border-radius:14px;padding:20px 24px;color:white;margin-bottom:15px;box-shadow:0 4px 18px rgba(11,35,69,.10);}
    .top-head h1 {margin:0;font-size:29px;color:white;}
    .top-head p {margin:5px 0 0;color:#d9e7fb;font-size:14px;}
    .section-title {font-size:17px;font-weight:750;color:#16345e;margin:3px 0 10px;display:flex;align-items:center;gap:8px;}
    .card {background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:15px 16px; box-shadow:0 2px 9px rgba(16,34,63,.04); min-height:108px;}
    .card-label {font-size:12px;color:var(--muted);font-weight:650;margin-bottom:8px;}
    .card-value {font-size:26px;color:#102f5f;font-weight:800;line-height:1.1;}
    .card-sub {font-size:11px;color:#7b8797;margin-top:6px;}
    .risk-high {background:#fff0f2;border-color:#ffd3da;}.risk-high .card-value{color:var(--bad)}
    .risk-medium {background:#fff7e9;border-color:#ffe1ad;}.risk-medium .card-value{color:var(--warn)}
    .risk-low {background:#eefaf5;border-color:#ccebdd;}.risk-low .card-value{color:var(--good)}
    .panel {background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px 15px;margin-bottom:12px;box-shadow:0 2px 8px rgba(16,34,63,.035);}
    .alert-bad {background:#fff1f2;border:1px solid #ffd2d7;color:#9e2534;border-radius:9px;padding:11px 13px;font-weight:700;margin-bottom:12px;}
    .alert-good {background:#eefaf5;border:1px solid #ccebdd;color:#116a49;border-radius:9px;padding:11px 13px;font-weight:700;margin-bottom:12px;}
    .small-note {font-size:11px;color:#7a8797;}
    .legend {display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#58667a;margin-top:8px;}
    .dot {width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:5px;}
    .scenario {background:#fff;border:1px solid var(--line);border-radius:10px;padding:11px 12px;min-height:92px;}
    .scenario-title {font-size:12px;font-weight:750;color:#17345f;}
    .scenario-num {font-size:20px;font-weight:800;color:#17345f;margin-top:8px;}
    .scenario-sub {font-size:10px;color:#768397;margin-top:3px;}
    div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:10px;overflow:hidden;}
    div[data-testid="stMetric"] {background:#fff;border:1px solid var(--line);padding:12px;border-radius:10px;}
    .stButton > button, .stDownloadButton > button {border-radius:8px;font-weight:700;}
    .stButton > button[kind="primary"] {background:#145fbf;border-color:#145fbf;}
    .footer-line {font-size:11px;color:#7b8797;padding-top:12px;border-top:1px solid var(--line);margin-top:14px;}
    @media (max-width: 900px){.block-container{padding-left:.8rem;padding-right:.8rem}.top-head h1{font-size:23px}.card-value{font-size:22px}}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_ml_model():
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(TRAINING[FEATURES], TRAINING["risk_class"])
    return model


MODEL = get_ml_model()


def get_secret(key, fallback):
    try:
        return st.secrets[key]
    except Exception:
        return fallback


def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


def login_page():
    st.markdown("<div style='height:8vh'></div>", unsafe_allow_html=True)
    left, centre, right = st.columns([1, 1.05, 1])
    with centre:
        st.markdown(
            """
            <div style="background:white;border:1px solid #dbe4ef;border-radius:16px;padding:26px 28px;box-shadow:0 12px 35px rgba(11,35,69,.08)">
              <div style="font-size:30px;font-weight:850;color:#0b2345">🏭 SURE-Plan</div>
              <div style="color:#667085;margin-top:5px">Production Decision Dashboard</div>
              <div style="font-size:12px;color:#8a95a6;margin-top:6px">AI/ML powered production planning & resource allocation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            if u == get_secret("ADMIN_USERNAME", "admin") and p == get_secret("ADMIN_PASSWORD", "admin2026"):
                st.session_state.authenticated = True
                st.session_state.role = "Admin"
                st.session_state.username = u
                st.rerun()
            elif u == get_secret("USER_USERNAME", "planner") and p == get_secret("USER_PASSWORD", "user2026"):
                st.session_state.authenticated = True
                st.session_state.role = "User"
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Admin controls benchmark assumptions. Planner/User enters today's actual operating conditions and orders.")


if not st.session_state.get("authenticated", False):
    login_page()
    st.stop()

role = st.session_state["role"]

if "settings" not in st.session_state:
    st.session_state.settings = {
        "base_capacity": 300.0,
        "benchmark_workers": 47,
        "max_overtime": 0.05,
        "company_name": "Shoe Upper Manufacturing Prototype",
    }

if "orders" not in st.session_state:
    st.session_state.orders = pd.DataFrame([
        {"Order": "A", "Quantity": 2000, "Due Date": pd.Timestamp("2026-09-30"), "Material Delay (days)": 0},
        {"Order": "B", "Quantity": 1500, "Due Date": pd.Timestamp("2026-09-15"), "Material Delay (days)": 0},
        {"Order": "C", "Quantity": 1100, "Due Date": pd.Timestamp("2026-09-10"), "Material Delay (days)": 0},
    ])

settings = st.session_state.settings

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        """
        <div class="sp-brand">
          <div class="sp-brand-icon">🏭</div>
          <div><div class="sp-brand-title">SURE-Plan</div><div class="sp-brand-sub">Smarter Decisions. Stable Production.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_options = ["Dashboard", "Reports"] + (["Admin Settings"] if role == "Admin" else [])
    page = st.radio("Navigation", nav_options, label_visibility="collapsed")
    st.divider()
    st.caption("SIGNED IN AS")
    st.write(f"**{st.session_state.get('username','planner')}**  ·  {role}")
    if st.button("Logout", use_container_width=True):
        logout()
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.caption("SURE-Plan v2.0")
    st.caption("Explainable AI + Operations Planning")


# ---------- Admin settings page ----------
if page == "Admin Settings":
    st.markdown(
        "<div class='top-head'><h1>Admin Configuration</h1><p>Set the factory benchmark values used by the planning model. These are not daily planner inputs.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.subheader("Factory benchmark settings")
        settings["company_name"] = st.text_input("Company / model label", settings["company_name"])
        settings["base_capacity"] = st.number_input(
            "Standard production capacity (uppers/day)", 50.0, 3000.0,
            float(settings["base_capacity"]), 10.0,
        )
        settings["benchmark_workers"] = st.number_input(
            "Standard production workforce", 1, 500,
            int(settings["benchmark_workers"]), 1,
        )
        settings["max_overtime"] = st.slider(
            "Maximum overtime capacity", 0.00, 0.30,
            float(settings["max_overtime"]), 0.01,
        )
        st.warning("Benchmark values are model/secondary assumptions unless independently verified with company records.")
    with c2:
        st.subheader("Model reference")
        st.json(settings)
        st.info(
            "Planner/User will enter Workers Present Today. The application automatically converts that headcount into a labour-availability factor using the standard workforce configured here."
        )

    st.subheader("Explainable Decision Tree")
    fig, ax = plt.subplots(figsize=(14, 7))
    plot_tree(
        MODEL,
        feature_names=FEATURES,
        class_names=[str(c) for c in MODEL.classes_],
        filled=False,
        rounded=True,
        fontsize=7,
        ax=ax,
    )
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Prototype validation is based on researcher-designed simulated data, not historical company validation.")
    st.stop()


# ---------- Inputs shared by Dashboard and Reports ----------
# Defaults/controls live in session state so changing pages does not reset them.
if "planning_date" not in st.session_state:
    st.session_state.planning_date = date(2026, 9, 1)
if "machine_pct" not in st.session_state:
    st.session_state.machine_pct = 100
if "workers_present" not in st.session_state:
    st.session_state.workers_present = int(settings["benchmark_workers"])

planning_date = st.session_state.planning_date
machine_availability = st.session_state.machine_pct / 100.0
workers_present = int(st.session_state.workers_present)
labour_availability = min(1.0, workers_present / max(int(settings["benchmark_workers"]), 1))


def calculate_plan():
    prepared = []
    prediction_rows = []
    for _, row in st.session_state.orders.iterrows():
        if pd.isna(row.get("Order")) or pd.isna(row.get("Quantity")) or pd.isna(row.get("Due Date")):
            continue

        order = str(row["Order"]).strip()
        qty = float(row["Quantity"])
        days = business_days_between(st.session_state.planning_date, row["Due Date"])
        mat_delay = int(row.get("Material Delay (days)", 0) or 0)
        effective_days = max(1, days - mat_delay)
        req_daily = qty / max(1, days)
        req_after_delay = qty / effective_days
        m = st.session_state.machine_pct / 100.0
        wp = int(st.session_state.workers_present)
        l = min(1.0, wp / max(int(settings["benchmark_workers"]), 1))
        effective_cap = settings["base_capacity"] * m * l
        ratio = req_after_delay / max(effective_cap, 1e-9)

        x = pd.DataFrame([{
            "quantity": qty,
            "days_remaining": days,
            "machine_availability": m,
            "labour_availability": l,
            "material_delay_days": mat_delay,
            "base_capacity": settings["base_capacity"],
            "required_daily_output": req_daily,
            "capacity_ratio": ratio,
        }])
        risk = str(MODEL.predict(x)[0])
        probs = MODEL.predict_proba(x)[0]
        conf = max(probs) * 100

        prepared.append({
            "order": order,
            "quantity": qty,
            "due_day": days,
            "days_remaining": days,
            "material_delay_days": mat_delay,
            "risk": risk,
            "ml_confidence": conf,
            "required_daily_output": req_daily,
            "capacity_ratio": ratio,
        })
        prediction_rows.append({
            "Order": order,
            "Quantity": int(qty),
            "Working Days Remaining": days,
            "Required Daily Output": round(req_daily, 1),
            "Capacity Ratio": round(ratio, 2),
            "ML Risk": risk,
            "Model Confidence %": round(conf, 1),
        })

    overtime = recommend_overtime(
        prepared,
        settings["base_capacity"],
        st.session_state.machine_pct / 100.0,
        min(1.0, int(st.session_state.workers_present) / max(int(settings["benchmark_workers"]), 1)),
        settings["max_overtime"],
    ) if prepared else 0.0

    results, daily, resource_info = simulate_orders(
        prepared,
        settings["base_capacity"],
        settings["benchmark_workers"],
        st.session_state.machine_pct / 100.0,
        min(1.0, int(st.session_state.workers_present) / max(int(settings["benchmark_workers"]), 1)),
        overtime,
    ) if prepared else (pd.DataFrame(), pd.DataFrame(), {})

    recommendations = build_recommendations(
        results,
        prepared,
        st.session_state.machine_pct / 100.0,
        min(1.0, int(st.session_state.workers_present) / max(int(settings["benchmark_workers"]), 1)),
        overtime,
    ) if not results.empty else pd.DataFrame()

    return prepared, pd.DataFrame(prediction_rows), overtime, results, daily, resource_info, recommendations


# ---------- Reports page ----------
if page == "Reports":
    prepared, pred_df, overtime, results, daily, resource_info, recommendations = calculate_plan()
    st.markdown(
        "<div class='top-head'><h1>Reports & Export</h1><p>Download the latest planning outputs and management summary.</p></div>",
        unsafe_allow_html=True,
    )
    if results.empty:
        st.info("Add valid orders on the Dashboard first.")
        st.stop()

    export_results = results.copy()
    export_results["projected_completion_date"] = export_results["completion_day"].apply(
        lambda x: add_business_days(st.session_state.planning_date, int(x)).date()
    )

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("⬇ Order Results CSV", export_results.to_csv(index=False).encode("utf-8"), "SURE_Plan_Order_Results.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇ Daily Allocation CSV", daily.to_csv(index=False).encode("utf-8"), "SURE_Plan_Daily_Allocation.csv", "text/csv", use_container_width=True)
    with d3:
        st.download_button("⬇ Recommendations CSV", recommendations.to_csv(index=False).encode("utf-8"), "SURE_Plan_Recommendations.csv", "text/csv", use_container_width=True)
    with d4:
        summary_text = f"""SURE-PLAN MANAGEMENT SUMMARY\n\nPlanning date: {st.session_state.planning_date}\nWorkers present today: {st.session_state.workers_present}\nStandard workforce: {settings['benchmark_workers']}\nMachine availability: {st.session_state.machine_pct}%\nDerived labour availability: {min(1.0, int(st.session_state.workers_present)/max(int(settings['benchmark_workers']),1)):.1%}\nStandard capacity: {settings['base_capacity']:.0f} uppers/day\nRecommended overtime: {overtime:.1%}\n\nOrders:\n{export_results.to_string(index=False)}\n\nRecommendations:\n{recommendations.to_string(index=False)}\n\nAcademic note:\nThe ML model is a proof-of-concept trained on researcher-designed simulated scenarios.\nBenchmark resource values are model/secondary-data assumptions unless independently verified.\n"""
        st.download_button("⬇ Management Summary TXT", summary_text.encode("utf-8"), "SURE_Plan_Management_Summary.txt", "text/plain", use_container_width=True)

    st.subheader("Order results")
    st.dataframe(export_results, use_container_width=True)
    st.subheader("Recommendations")
    st.dataframe(recommendations, use_container_width=True)
    st.stop()


# ---------- Dashboard page ----------
st.markdown(
    f"""
    <div class="top-head">
      <h1>Production Decision Dashboard</h1>
      <p>AI/ML-driven planning, resource allocation and delivery-risk analysis in one management view · Signed in as {role}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-title'>1. Enter Order Details & Current Conditions</div>", unsafe_allow_html=True)
with st.container(border=True):
    t1, t2, t3 = st.columns([1, 1, 1])
    with t1:
        st.session_state.planning_date = st.date_input("Planning Date", value=st.session_state.planning_date)
    with t2:
        st.session_state.workers_present = st.number_input(
            "Workers Present Today",
            min_value=1,
            max_value=max(int(settings["benchmark_workers"]), 1),
            value=min(int(st.session_state.workers_present), int(settings["benchmark_workers"])),
            step=1,
            help="Enter the actual number of production workers present today."
        )
    with t3:
        st.session_state.machine_pct = st.slider(
            "Machine Availability Today",
            min_value=50,
            max_value=100,
            value=int(st.session_state.machine_pct),
            step=1,
            format="%d%%",
        )

    st.caption(
        f"Admin reference (read only): Standard workforce {int(settings['benchmark_workers'])} workers · "
        f"Standard capacity {settings['base_capacity']:.0f} uppers/day · Max overtime {settings['max_overtime']:.0%}"
    )

    st.markdown("**Active Orders**")
    st.session_state.orders = st.data_editor(
        st.session_state.orders,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Order": st.column_config.TextColumn("Order ID", required=True),
            "Quantity": st.column_config.NumberColumn("Order Quantity", min_value=1, step=50, required=True),
            "Due Date": st.column_config.DateColumn("Due Date", required=True),
            "Material Delay (days)": st.column_config.NumberColumn("Material Delay", min_value=0, step=1),
        },
    )
    run = st.button("▶ Run SURE-Plan Analysis", type="primary", use_container_width=True)
    if run:
        st.toast("SURE-Plan analysis updated using today's actual conditions.")

prepared, pred_df, overtime, results, daily, resource_info, recommendations = calculate_plan()

if results.empty:
    st.info("Enter at least one valid order to generate the management dashboard.")
    st.stop()

# Overall risk based on worst risk class / lateness
risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
worst_risk = max((str(r).upper() for r in results["risk"]), key=lambda x: risk_rank.get(x, 0))
late_count = int((results["on_time"] == "NO").sum())
if late_count > 0 and worst_risk == "LOW":
    worst_risk = "MEDIUM"
risk_class = "risk-high" if worst_risk == "HIGH" else ("risk-medium" if worst_risk == "MEDIUM" else "risk-low")

labour_availability = min(1.0, int(st.session_state.workers_present) / max(int(settings["benchmark_workers"]), 1))
effective_capacity = resource_info.get("effective_daily_capacity", 0)
total_tardiness = int(results["tardiness_days"].sum())
on_time = int((results["on_time"] == "YES").sum())

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='card {risk_class}'><div class='card-label'>Overall Risk</div><div class='card-value'>{worst_risk}</div><div class='card-sub'>{late_count} order(s) currently projected late</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><div class='card-label'>Available Capacity Today</div><div class='card-value'>{effective_capacity:.0f}</div><div class='card-sub'>uppers/day after attendance, machine & overtime</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><div class='card-label'>Workers Present</div><div class='card-value'>{int(st.session_state.workers_present)}</div><div class='card-sub'>{labour_availability:.1%} of standard workforce</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><div class='card-label'>Orders On Time</div><div class='card-value'>{on_time}/{len(results)}</div><div class='card-sub'>under current production plan</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='card'><div class='card-label'>Projected Tardiness</div><div class='card-value'>{total_tardiness}</div><div class='card-sub'>total working days across active orders</div></div>", unsafe_allow_html=True)

st.write("")
left, right = st.columns([1.05, 1])
with left:
    st.markdown("<div class='section-title'>2. AI/ML Risk Assessment</div>", unsafe_allow_html=True)
    display_pred = pred_df.copy()
    display_pred = display_pred[["Order", "Quantity", "Required Daily Output", "Capacity Ratio", "ML Risk", "Model Confidence %"]]
    st.dataframe(display_pred, use_container_width=True, hide_index=True, height=225)
    st.markdown(
        "<div class='legend'><span><span class='dot' style='background:#16845b'></span>LOW</span><span><span class='dot' style='background:#e18b14'></span>MEDIUM</span><span><span class='dot' style='background:#d63f50'></span>HIGH</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Model confidence is a prototype decision-tree probability from simulated training data; it is not certainty about the real delivery outcome.")

with right:
    st.markdown("<div class='section-title'>3. Resource Allocation Summary</div>", unsafe_allow_html=True)
    alloc = results[["order", "day1_workers", "completion_day", "tardiness_days", "on_time"]].copy()
    alloc.columns = ["Order", "Recommended Workers", "Projected Completion Day", "Tardiness (Days)", "On Time?"]
    st.dataframe(alloc, use_container_width=True, hide_index=True, height=225)
    st.caption(f"Recommended overtime: {overtime:.1%}. Allocation prioritizes due-date pressure, capacity shortfall and ML risk while respecting today's worker availability.")

left2, right2 = st.columns([1.05, 1])
with left2:
    st.markdown("<div class='section-title'>4. Daily Resource Allocation</div>", unsafe_allow_html=True)
    if not daily.empty:
        pivot = daily.pivot(index="day", columns="order", values="workers_allocated").fillna(0)
        st.line_chart(pivot, height=310)
        st.caption("Worker allocation changes automatically as material becomes available and orders are completed.")

with right2:
    st.markdown("<div class='section-title'>5. Management Recommendation</div>", unsafe_allow_html=True)
    if late_count:
        st.markdown(f"<div class='alert-bad'>⚠ {late_count} of {len(results)} active order(s) are projected LATE under current conditions.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-good'>✓ Current resource plan is projected to complete all active orders on time.</div>", unsafe_allow_html=True)

    if not recommendations.empty:
        # Show concise management actions, highest risk / lateness first.
        merged = recommendations.merge(results[["order", "tardiness_days"]], on="order", how="left")
        merged["risk_rank"] = merged["risk"].map(risk_rank).fillna(0)
        merged = merged.sort_values(["tardiness_days", "risk_rank"], ascending=False)
        for _, row in merged.iterrows():
            icon = "🔴" if row["risk"] == "HIGH" else ("🟠" if row["risk"] == "MEDIUM" else "🟢")
            st.markdown(f"**{icon} Order {row['order']} · {row['risk']} Risk · {row['status']}**")
            st.write(row["recommended_action"])
    st.info(f"Management control: up to {settings['max_overtime']:.0%} overtime is permitted by the current Admin setting. If lateness persists, compare capacity recovery, material expediting, subcontracting and due-date renegotiation.")

# What-if scenarios
st.markdown("<div class='section-title'>6. Quick View — What-if Scenarios</div>", unsafe_allow_html=True)
scenario_defs = [
    ("Normal Resources", 1.00, 1.00, 0),
    ("Machine Breakdown", 0.80, 1.00, 0),
    ("Labour Shortage", 1.00, 0.80, 0),
    ("Combined Disruption", 0.85, 0.90, 2),
    ("Severe Combined", 0.75, 0.80, 4),
]
scenario_output = []
for name, m, l, c_delay in scenario_defs:
    temp_orders = []
    for o in prepared:
        x_o = dict(o)
        if x_o["order"].upper() == "C":
            x_o["material_delay_days"] = max(x_o["material_delay_days"], c_delay)
        effective_days = max(1, x_o["days_remaining"] - x_o["material_delay_days"])
        req_daily = x_o["quantity"] / max(1, x_o["days_remaining"])
        ratio = (x_o["quantity"] / effective_days) / max(settings["base_capacity"] * m * l, 1e-9)
        x = pd.DataFrame([{
            "quantity": x_o["quantity"],
            "days_remaining": x_o["days_remaining"],
            "machine_availability": m,
            "labour_availability": l,
            "material_delay_days": x_o["material_delay_days"],
            "base_capacity": settings["base_capacity"],
            "required_daily_output": req_daily,
            "capacity_ratio": ratio,
        }])
        risk = str(MODEL.predict(x)[0])
        probs = MODEL.predict_proba(x)[0]
        x_o["risk"] = risk
        x_o["ml_confidence"] = max(probs) * 100
        x_o["required_daily_output"] = req_daily
        x_o["capacity_ratio"] = ratio
        temp_orders.append(x_o)

    sc_ot = recommend_overtime(temp_orders, settings["base_capacity"], m, l, settings["max_overtime"]) if temp_orders else 0
    sc_res, _, _ = simulate_orders(temp_orders, settings["base_capacity"], settings["benchmark_workers"], m, l, sc_ot) if temp_orders else (pd.DataFrame(), pd.DataFrame(), {})
    scenario_output.append({
        "Scenario": name,
        "On Time": int((sc_res["on_time"] == "YES").sum()) if not sc_res.empty else 0,
        "Total": len(sc_res),
        "Tardiness": int(sc_res["tardiness_days"].sum()) if not sc_res.empty else 0,
        "Machine": m,
        "Labour": l,
    })

sc_cols = st.columns(len(scenario_output))
for col, sc in zip(sc_cols, scenario_output):
    with col:
        st.markdown(
            f"<div class='scenario'><div class='scenario-title'>{sc['Scenario']}</div><div class='scenario-num'>{sc['On Time']} / {sc['Total']}</div><div class='scenario-sub'>On time · {sc['Tardiness']} total tardiness days<br>{sc['Machine']:.0%} machine · {sc['Labour']:.0%} labour</div></div>",
            unsafe_allow_html=True,
        )

with st.expander("Management detail: order-level reasons and full schedule"):
    st.subheader("Order-level recommendations")
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.subheader("Full daily allocation")
    st.dataframe(daily, use_container_width=True, hide_index=True, height=340)

st.markdown(
    "<div class='footer-line'>SURE-Plan MBA prototype · Explainable ML + disruption-aware operations planning · Academic note: ML training scenarios are researcher-designed/simulated and should not be presented as validated historical company data.</div>",
    unsafe_allow_html=True,
)
