from pathlib import Path
from datetime import date
import copy
import math
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from planner_core import (
    business_days_between,
    business_days_until_ready,
    add_business_days,
    recommend_overtime,
    simulate_orders,
    build_recommendations,
)

BASE = Path(__file__).resolve().parent
TRAINING = pd.read_csv(BASE / "prototype_training_data.csv")
FEATURES = [
    "quantity",
    "days_remaining",
    "machine_availability",
    "labour_availability",
    "material_delay_days",
    "base_capacity",
    "required_daily_output",
    "capacity_ratio",
]

MATERIAL_STATUSES = [
    "Ready",
    "Partially Ready",
    "Awaiting Material",
    "Quality Hold",
    "Shortage",
    "Rejected / Replacement Required",
]
BLOCKED_MATERIAL_STATUSES = {
    "Awaiting Material",
    "Quality Hold",
    "Shortage",
    "Rejected / Replacement Required",
}

DEFAULT_MACHINE_MASTER = pd.DataFrame(
    [
        {"Sequence": 1, "Process": "Material Cutting / Clicking", "Machine Type": "Cutting / Clicking Press", "Total Machines": 8},
        {"Sequence": 2, "Process": "Skiving", "Machine Type": "Skiving Machine", "Total Machines": 10},
        {"Sequence": 3, "Process": "Splitting", "Machine Type": "Splitting Machine", "Total Machines": 3},
        {"Sequence": 4, "Process": "Marking / Stamping", "Machine Type": "Marking / Stamping Machine", "Total Machines": 4},
        {"Sequence": 5, "Process": "Folding / Edge Preparation", "Machine Type": "Folding / Edge Preparation Machine", "Total Machines": 8},
        {"Sequence": 6, "Process": "Sewing / Stitching", "Machine Type": "Sewing / Stitching Machine", "Total Machines": 45},
        {"Sequence": 7, "Process": "Zig-zag / Special Stitch", "Machine Type": "Zig-zag / Special Stitch Machine", "Total Machines": 6},
        {"Sequence": 8, "Process": "Bartack / Reinforcement", "Machine Type": "Bartack / Reinforcement Machine", "Total Machines": 4},
        {"Sequence": 9, "Process": "Eyelet / Punching", "Machine Type": "Eyelet / Punching Machine", "Total Machines": 5},
        {"Sequence": 10, "Process": "Fusing / Heat Press", "Machine Type": "Fusing / Heat Press", "Total Machines": 5},
        {"Sequence": 11, "Process": "Final Upper Preparation / Support", "Machine Type": "Preparation / Support Equipment", "Total Machines": 7},
    ]
)

st.set_page_config(
    page_title="SURE-Plan | Production Decision Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Professional dashboard styling - preserves the user's navy + gradient theme.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy:#061743;
        --navy2:#0a2c68;
        --blue:#1769e0;
        --violet:#7045f3;
        --magenta:#ed2aa7;
        --cyan:#0fa9bd;
        --ink:#11264e;
        --muted:#66738f;
        --line:#e5e9f3;
        --panel:#ffffff;
        --good:#139b66;
        --warn:#e69412;
        --bad:#ef476f;
        --soft:#f7f9fd;
    }
    .stApp {
        color:var(--ink);
        background:
          radial-gradient(circle at 17% 2%, rgba(237,42,167,.075), transparent 23%),
          radial-gradient(circle at 76% 0%, rgba(23,105,224,.075), transparent 24%),
          linear-gradient(180deg,#fbfbfe 0%,#f7f9fd 55%,#fbfcff 100%);
    }
    [data-testid="stHeader"] {background:rgba(250,251,254,.90);backdrop-filter:blur(10px);}
    [data-testid="stSidebar"] {
        background:
          radial-gradient(circle at 25% 82%, rgba(112,69,243,.31), transparent 28%),
          linear-gradient(180deg,#061743 0%,#08265d 55%,#061743 100%);
        border-right:1px solid rgba(255,255,255,.07);
    }
    [data-testid="stSidebar"] * {color:#f6f8ff;}
    [data-testid="stSidebar"] .stRadio label {padding:.56rem .72rem;border-radius:10px;transition:.18s ease;}
    [data-testid="stSidebar"] .stRadio label:hover {background:rgba(255,255,255,.08);}
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background:linear-gradient(100deg,#6c45f2 0%,#bf36de 58%,#ec309e 100%);
        box-shadow:0 8px 22px rgba(129,61,224,.28);
    }
    .block-container {padding-top:.85rem;padding-bottom:2rem;max-width:1600px;}
    h1,h2,h3 {letter-spacing:-.025em;color:#10234b;}
    .sp-brand {display:flex;align-items:center;gap:12px;margin:4px 0 18px;}
    .sp-brand-icon {
        width:44px;height:44px;border-radius:13px;
        background:linear-gradient(145deg,#2d69ff,#7447f3 50%,#ec2aa8);
        display:flex;align-items:center;justify-content:center;font-size:23px;
        box-shadow:0 9px 22px rgba(86,68,240,.30);
    }
    .sp-brand-title {font-size:25px;font-weight:820;line-height:1;color:white;}
    .sp-brand-sub {font-size:10.5px;opacity:.78;margin-top:5px;}
    .top-head {
        background:rgba(255,255,255,.92);border:1px solid var(--line);border-radius:15px;
        padding:17px 21px;margin-bottom:13px;box-shadow:0 7px 22px rgba(28,46,88,.055);
        position:relative;overflow:hidden;
    }
    .top-head:after {content:"";position:absolute;right:-65px;top:-90px;width:250px;height:230px;border-radius:50%;background:radial-gradient(circle,rgba(237,42,167,.10),rgba(112,69,243,.05) 45%,transparent 70%);}
    .top-head h1 {margin:0;font-size:28px;color:#0d2b68;position:relative;z-index:1;}
    .top-head p {margin:4px 0 0;color:#596a8b;font-size:13px;position:relative;z-index:1;}
    .section-title {font-size:16.5px;font-weight:800;color:#12316b;margin:4px 0 9px;display:flex;align-items:center;gap:8px;}
    .section-sub {font-size:11.5px;color:#748198;margin:-4px 0 9px;}
    .card,.scenario,div[data-testid="stMetric"] {
        background:rgba(255,255,255,.95);border:1px solid var(--line);
        box-shadow:0 7px 20px rgba(28,46,88,.052);
    }
    .card {border-radius:14px;padding:14px 15px;min-height:112px;position:relative;overflow:hidden;}
    .card:after {content:"";position:absolute;left:0;right:0;bottom:0;height:3px;background:linear-gradient(90deg,#1769e0,#7045f3,#ed2aa7);opacity:.20;}
    .card-label {font-size:11.5px;color:var(--muted);font-weight:700;margin-bottom:7px;}
    .card-value {font-size:25px;color:#102f68;font-weight:840;line-height:1.1;}
    .card-sub {font-size:10.5px;color:#7b879f;margin-top:6px;line-height:1.35;}
    .risk-high {background:linear-gradient(145deg,#fff5f8,#fff);border-color:#ffd6e1;}.risk-high .card-value{color:var(--bad)}
    .risk-medium {background:linear-gradient(145deg,#fff9ef,#fff);border-color:#ffe2af;}.risk-medium .card-value{color:#c97900}
    .risk-low {background:linear-gradient(145deg,#f1fff9,#fff);border-color:#c9ecdc;}.risk-low .card-value{color:var(--good)}
    .scenario {border-radius:12px;padding:11px 12px;min-height:108px;}
    .scenario-title {font-size:11.5px;font-weight:800;color:#17345f;line-height:1.25;}
    .scenario-badge {font-size:10px;font-weight:800;margin-top:5px;}
    .scenario-num {font-size:20px;font-weight:840;color:#17345f;margin-top:6px;}
    .scenario-sub {font-size:10px;color:#758297;margin-top:3px;line-height:1.35;}
    .alert-bad {background:#fff2f6;border:1px solid #ffd8e2;color:#9f2948;border-radius:10px;padding:11px 13px;font-weight:720;margin-bottom:10px;}
    .alert-good {background:#effcf7;border:1px solid #caeedf;color:#0d7550;border-radius:10px;padding:11px 13px;font-weight:720;margin-bottom:10px;}
    .alert-warn {background:#fff9ef;border:1px solid #ffe1ae;color:#9a6400;border-radius:10px;padding:11px 13px;font-weight:720;margin-bottom:10px;}
    .pill-row {display:flex;gap:8px;flex-wrap:wrap;margin:7px 0 2px;}
    .pill {padding:4px 8px;border-radius:999px;background:#f2f5fb;color:#50617f;font-size:10.5px;font-weight:700;border:1px solid #e3e8f1;}
    div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:0 6px 16px rgba(28,46,88,.035);}
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div,
    [data-testid="stNumberInput"] input,[data-testid="stDateInput"] input {background:#fff !important;border-color:#e0e6f0 !important;border-radius:10px !important;}
    .stButton > button,.stDownloadButton > button {border-radius:10px;font-weight:760;min-height:2.55rem;}
    .stButton > button[kind="primary"] {
        background:linear-gradient(95deg,#0b9bd8 0%,#236af0 35%,#7546ef 67%,#ed2aa7 100%);
        border:0;box-shadow:0 7px 18px rgba(92,65,226,.20);
    }
    .stButton > button[kind="primary"]:hover {filter:brightness(1.04);transform:translateY(-1px);}
    .footer-line {font-size:10.5px;color:#7a8799;padding-top:11px;border-top:1px solid var(--line);margin-top:14px;}
    .mini-label {font-size:10px;color:#7b8797;text-transform:uppercase;letter-spacing:.06em;font-weight:800;}
    @media (max-width:900px){.block-container{padding-left:.8rem;padding-right:.8rem}.top-head h1{font-size:23px}.card-value{font-size:22px}}
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
    _, centre, _ = st.columns([1, 1.05, 1])
    with centre:
        st.markdown(
            """
            <div style="background:linear-gradient(145deg,rgba(255,255,255,.99),rgba(250,247,255,.98));border:1px solid #e5e6f0;border-radius:18px;padding:28px 30px;box-shadow:0 18px 45px rgba(70,45,140,.10)">
              <div style="font-size:30px;font-weight:850;color:#122d63">⬢ SURE-Plan</div>
              <div style="color:#667085;margin-top:5px">Production Decision Dashboard</div>
              <div style="font-size:12px;color:#8a95a6;margin-top:6px">Explainable AI + rolling operations planning</div>
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
                st.session_state.role = "Planner"
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.caption("Admin controls benchmark assumptions. Planner enters today's actual operating conditions and active orders.")


if not st.session_state.get("authenticated", False):
    login_page()
    st.stop()

role = st.session_state["role"]

# -----------------------------------------------------------------------------
# Session-state defaults
# -----------------------------------------------------------------------------
if "settings" not in st.session_state:
    st.session_state.settings = {
        "base_capacity": 300.0,
        "benchmark_workers": 150,
        "max_overtime": 0.05,
        "company_name": "Shoe Upper Manufacturing Prototype",
    }

if "machine_master" not in st.session_state:
    st.session_state.machine_master = DEFAULT_MACHINE_MASTER.copy()

if "planning_date" not in st.session_state:
    st.session_state.planning_date = date(2026, 9, 1)

if "workers_present" not in st.session_state:
    st.session_state.workers_present = int(st.session_state.settings["benchmark_workers"])

if "orders" not in st.session_state:
    processes = DEFAULT_MACHINE_MASTER["Process"].tolist()
    st.session_state.orders = pd.DataFrame(
        [
            {
                "Order": "A",
                "Original Quantity": 2000,
                "Completed Before Today": 0,
                "Actual Production Today": 0,
                "Workers Used Today": 0,
                "Due Date": pd.Timestamp("2026-09-30"),
                "Material Status": "Ready",
                "Expected Material Ready Date": pd.NaT,
                "Current Process": processes[0],
            },
            {
                "Order": "B",
                "Original Quantity": 1500,
                "Completed Before Today": 0,
                "Actual Production Today": 0,
                "Workers Used Today": 0,
                "Due Date": pd.Timestamp("2026-09-15"),
                "Material Status": "Ready",
                "Expected Material Ready Date": pd.NaT,
                "Current Process": "Sewing / Stitching",
            },
            {
                "Order": "C",
                "Original Quantity": 1100,
                "Completed Before Today": 0,
                "Actual Production Today": 0,
                "Workers Used Today": 0,
                "Due Date": pd.Timestamp("2026-09-10"),
                "Material Status": "Ready",
                "Expected Material Ready Date": pd.NaT,
                "Current Process": "Folding / Edge Preparation",
            },
        ]
    )

if "machine_today" not in st.session_state:
    st.session_state.machine_today = pd.DataFrame()

if "plan_history" not in st.session_state:
    st.session_state.plan_history = []

settings = st.session_state.settings


def clean_machine_master(df):
    expected = ["Sequence", "Process", "Machine Type", "Total Machines"]
    out = df.copy()
    for col in expected:
        if col not in out.columns:
            out[col] = None
    out = out[expected]
    out["Process"] = out["Process"].fillna("").astype(str).str.strip()
    out["Machine Type"] = out["Machine Type"].fillna("").astype(str).str.strip()
    out["Total Machines"] = pd.to_numeric(out["Total Machines"], errors="coerce").fillna(0).clip(lower=0).round().astype(int)
    out["Sequence"] = pd.to_numeric(out["Sequence"], errors="coerce")
    missing_seq = out["Sequence"].isna()
    if missing_seq.any():
        start = int(out["Sequence"].dropna().max()) + 1 if out["Sequence"].notna().any() else 1
        out.loc[missing_seq, "Sequence"] = range(start, start + int(missing_seq.sum()))
    out["Sequence"] = out["Sequence"].astype(int)
    out = out[(out["Process"] != "") & (out["Total Machines"] >= 0)].copy()
    out = out.sort_values("Sequence").reset_index(drop=True)
    return out


def sync_machine_today():
    master = clean_machine_master(st.session_state.machine_master)
    old = st.session_state.machine_today.copy() if isinstance(st.session_state.machine_today, pd.DataFrame) else pd.DataFrame()
    old_lookup = {}
    if not old.empty and "Process" in old.columns:
        for _, r in old.iterrows():
            old_lookup[str(r.get("Process", ""))] = r

    rows = []
    for _, r in master.iterrows():
        process = r["Process"]
        total = int(r["Total Machines"])
        prev = old_lookup.get(process)
        if prev is None:
            available = total
            issue = ""
        else:
            available = int(pd.to_numeric(prev.get("Available Today", total), errors="coerce") if pd.notna(prev.get("Available Today", total)) else total)
            issue = str(prev.get("Breakdown / Issue", "") or "")
        available = max(0, min(total, available))
        rows.append({
            "Sequence": int(r["Sequence"]),
            "Process": process,
            "Machine Type": r["Machine Type"],
            "Total Machines": total,
            "Available Today": available,
            "Breakdown / Issue": issue,
        })
    st.session_state.machine_today = pd.DataFrame(rows)


def machine_status_table(machine_today=None):
    df = (machine_today if machine_today is not None else st.session_state.machine_today).copy()
    if df.empty:
        return df
    df["Total Machines"] = pd.to_numeric(df["Total Machines"], errors="coerce").fillna(0).astype(int)
    df["Available Today"] = pd.to_numeric(df["Available Today"], errors="coerce").fillna(0).astype(int)
    df["Available Today"] = [max(0, min(int(t), int(a))) for t, a in zip(df["Total Machines"], df["Available Today"])]
    df["Breakdown / Unavailable"] = (df["Total Machines"] - df["Available Today"]).clip(lower=0)
    df["Availability %"] = [
        (a / t * 100.0) if t > 0 else 100.0
        for a, t in zip(df["Available Today"], df["Total Machines"])
    ]
    def status(pct):
        if pct >= 90:
            return "🟢 Stable"
        if pct >= 75:
            return "🟠 Warning"
        return "🔴 Critical"
    df["Status"] = df["Availability %"].map(status)
    return df


def machine_availability_map(machine_today=None):
    df = machine_status_table(machine_today)
    if df.empty:
        return {}, 1.0, "No machine master", 0, 0, 1.0
    amap = {
        str(r["Process"]): (float(r["Available Today"]) / float(r["Total Machines"]) if int(r["Total Machines"]) > 0 else 1.0)
        for _, r in df.iterrows()
    }
    valid = [(p, v) for p, v in amap.items() if v >= 0]
    if valid and min(v for _, v in valid) >= 0.999999:
        bottleneck_process, bottleneck_factor = "No current bottleneck", 1.0
    elif valid:
        min_factor = min(v for _, v in valid)
        tied = df[df["Process"].map(amap).sub(min_factor).abs() < 1e-9].sort_values("Total Machines", ascending=False)
        bottleneck_process = str(tied.iloc[0]["Process"]) if not tied.empty else min(valid, key=lambda x: x[1])[0]
        bottleneck_factor = min_factor
    else:
        bottleneck_process, bottleneck_factor = "No process", 1.0
    total = int(df["Total Machines"].sum())
    available = int(df["Available Today"].sum())
    overall = (available / total) if total > 0 else 1.0
    return amap, bottleneck_factor, bottleneck_process, total, available, overall


def process_options():
    master = clean_machine_master(st.session_state.machine_master)
    opts = master["Process"].tolist()
    return opts if opts else ["General Production"]


def normalize_orders(df):
    cols = [
        "Order",
        "Original Quantity",
        "Completed Before Today",
        "Actual Production Today",
        "Workers Used Today",
        "Due Date",
        "Material Status",
        "Expected Material Ready Date",
        "Current Process",
    ]
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols]
    out["Order"] = out["Order"].fillna("").astype(str).str.strip()
    for c in ["Original Quantity", "Completed Before Today", "Actual Production Today", "Workers Used Today"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(lower=0)
    out["Original Quantity"] = out["Original Quantity"].round().astype(int)
    out["Completed Before Today"] = out["Completed Before Today"].round().astype(int)
    out["Actual Production Today"] = out["Actual Production Today"].round().astype(int)
    out["Workers Used Today"] = out["Workers Used Today"].round().astype(int)
    out["Due Date"] = pd.to_datetime(out["Due Date"], errors="coerce")
    out["Expected Material Ready Date"] = pd.to_datetime(out["Expected Material Ready Date"], errors="coerce")
    out["Material Status"] = out["Material Status"].where(out["Material Status"].isin(MATERIAL_STATUSES), "Ready")
    opts = process_options()
    out["Current Process"] = out["Current Process"].where(out["Current Process"].isin(opts), opts[0])
    max_completed = out["Original Quantity"]
    total_done = out["Completed Before Today"] + out["Actual Production Today"]
    over = total_done > max_completed
    if over.any():
        out.loc[over, "Actual Production Today"] = (
            out.loc[over, "Original Quantity"] - out.loc[over, "Completed Before Today"]
        ).clip(lower=0)
    return out


def material_delay_info(row, planning_date_value, status_override=None, extra_delay=0):
    status = status_override or str(row.get("Material Status", "Ready"))
    expected = row.get("Expected Material Ready Date")
    planning_delay = 0
    if status in BLOCKED_MATERIAL_STATUSES:
        if pd.isna(expected):
            # Keep the order effectively blocked in the operations simulation until a ready date is entered.
            planning_delay = 120
        else:
            planning_delay = max(1, business_days_until_ready(planning_date_value, expected))
    planning_delay += max(0, int(extra_delay))
    # Training scenarios only cover 0-6 material-delay days. Keep ML input in-range and disclose it.
    ml_delay = min(6, planning_delay)
    eligible_today = status not in BLOCKED_MATERIAL_STATUSES and planning_delay == 0
    if status == "Partially Ready":
        eligible_today = True
    return status, planning_delay, ml_delay, eligible_today


def predict_risk(feature_row):
    x = pd.DataFrame([feature_row], columns=FEATURES)
    risk = str(MODEL.predict(x)[0]).upper()
    probs = MODEL.predict_proba(x)[0]
    confidence = float(max(probs) * 100.0)
    return risk, confidence


def risk_drivers(order, labour_factor):
    drivers = []
    status = order["material_status"]
    if status in BLOCKED_MATERIAL_STATUSES:
        drivers.append(f"Material: {status}")
    elif status == "Partially Ready":
        drivers.append("Material partially ready")
    if order["machine_availability"] < 0.90:
        drivers.append(f"{order['current_process']} machines {order['machine_availability']:.0%}")
    if labour_factor < 0.90:
        drivers.append(f"Labour availability {labour_factor:.0%}")
    if order["days_remaining"] <= 5:
        drivers.append(f"Only {order['days_remaining']} working day(s) remaining")
    if order["capacity_ratio"] >= 1.0:
        drivers.append("Required output exceeds current capacity basis")
    elif order["capacity_ratio"] >= 0.85:
        drivers.append("Capacity buffer is tight")
    if not drivers:
        drivers.append("Current capacity buffer is manageable")
    return " • ".join(drivers[:4])


def calculate_plan(
    orders_source=None,
    workers_override=None,
    machine_map_override=None,
    extra_delay_map=None,
    status_override_map=None,
    overtime_override=None,
):
    orders_df = normalize_orders(orders_source if orders_source is not None else st.session_state.orders)
    workers = int(st.session_state.workers_present if workers_override is None else workers_override)
    workforce = max(1, int(settings["benchmark_workers"]))
    # ML training range stops at 100%; operationally this avoids claiming linear gains above the standard workforce.
    labour_factor = max(0.0, min(1.0, workers / workforce))

    current_map, current_bottleneck, _, _, _, _ = machine_availability_map()
    m_map = dict(current_map if machine_map_override is None else machine_map_override)
    aggregate_machine = min(m_map.values()) if m_map else 1.0
    extra_delay_map = extra_delay_map or {}
    status_override_map = status_override_map or {}

    prepared = []
    prediction_rows = []
    for _, row in orders_df.iterrows():
        if not str(row.get("Order", "")).strip() or pd.isna(row.get("Due Date")):
            continue
        original_qty = int(row.get("Original Quantity", 0) or 0)
        completed = int(row.get("Completed Before Today", 0) or 0) + int(row.get("Actual Production Today", 0) or 0)
        remaining_qty = max(0, original_qty - completed)
        if original_qty <= 0 or remaining_qty <= 0:
            continue

        order_id = str(row["Order"]).strip()
        days = business_days_between(st.session_state.planning_date, row["Due Date"])
        process = str(row.get("Current Process", process_options()[0]))
        process_machine = max(0.0, min(1.0, float(m_map.get(process, aggregate_machine))))
        status, planning_delay, ml_delay, eligible_today = material_delay_info(
            row,
            st.session_state.planning_date,
            status_override=status_override_map.get(order_id),
            extra_delay=extra_delay_map.get(order_id, 0),
        )
        effective_days = max(1, days - min(planning_delay, max(days - 1, 0)))
        req_daily = remaining_qty / max(1, days)
        req_after_delay = remaining_qty / effective_days
        effective_capacity = float(settings["base_capacity"]) * process_machine * labour_factor
        ratio = req_after_delay / max(effective_capacity, 1e-9)

        feature_row = {
            "quantity": float(remaining_qty),
            "days_remaining": int(days),
            "machine_availability": float(process_machine),
            "labour_availability": float(labour_factor),
            "material_delay_days": int(ml_delay),
            "base_capacity": float(settings["base_capacity"]),
            "required_daily_output": float(req_daily),
            "capacity_ratio": float(ratio),
        }
        risk, confidence = predict_risk(feature_row)
        prepared_order = {
            "order": order_id,
            "original_quantity": original_qty,
            "quantity": remaining_qty,
            "due_day": days,
            "days_remaining": days,
            "material_delay_days": planning_delay,
            "ml_material_delay_days": ml_delay,
            "material_status": status,
            "eligible_today": eligible_today,
            "expected_material_ready": row.get("Expected Material Ready Date"),
            "current_process": process,
            "machine_availability": process_machine,
            "risk": risk,
            "ml_confidence": confidence,
            "required_daily_output": req_daily,
            "capacity_ratio": ratio,
        }
        prepared_order["risk_drivers"] = risk_drivers(prepared_order, labour_factor)
        prepared.append(prepared_order)
        prediction_rows.append({
            "Order": order_id,
            "Remaining Quantity": int(remaining_qty),
            "Required Daily Output": round(req_daily, 1),
            "Capacity Ratio": round(ratio, 2),
            "ML Delivery Risk": risk,
            "Prototype Model Confidence": round(confidence, 1),
            "Key Drivers": prepared_order["risk_drivers"],
        })

    active_machine_factor = min((o["machine_availability"] for o in prepared), default=aggregate_machine)

    if overtime_override is None:
        overtime = recommend_overtime(
            prepared,
            settings["base_capacity"],
            active_machine_factor,
            labour_factor,
            settings["max_overtime"],
        ) if prepared else 0.0
    else:
        overtime = max(0.0, min(float(settings["max_overtime"]), float(overtime_override)))

    results, daily, resource_info = simulate_orders(
        prepared,
        settings["base_capacity"],
        settings["benchmark_workers"],
        labour_factor,
        overtime,
    ) if prepared else (pd.DataFrame(), pd.DataFrame(), {})

    recommendations = build_recommendations(
        results,
        prepared,
        labour_factor,
        overtime,
    ) if not results.empty else pd.DataFrame()

    if not daily.empty:
        daily = daily.copy()
        daily["production_date"] = daily["day"].apply(
            lambda d: add_business_days(st.session_state.planning_date, int(d) - 1).date()
        )

    resource_info = dict(resource_info)
    resource_info.update({
        "labour_availability": labour_factor,
        "aggregate_machine_factor": active_machine_factor,
        "effective_daily_capacity": float(settings["base_capacity"]) * labour_factor * active_machine_factor * (1.0 + overtime),
        "workers_present": workers,
    })
    return prepared, pd.DataFrame(prediction_rows), overtime, results, daily, resource_info, recommendations


def build_material_tracker(prepared):
    rows = []
    for o in prepared:
        expected = o.get("expected_material_ready")
        if pd.isna(expected):
            expected_text = "—" if o["material_status"] in {"Ready", "Partially Ready"} else "Ready date required"
        else:
            expected_text = pd.Timestamp(expected).strftime("%d %b %Y")
        if o["material_status"] == "Ready":
            action = "No material action"
        elif o["material_status"] == "Partially Ready":
            action = "Confirm next-stage component readiness"
        elif o["material_status"] == "Quality Hold":
            action = "Expedite inspection / release / replacement"
        elif o["material_status"] == "Rejected / Replacement Required":
            action = "Arrange replacement before releasing production"
        else:
            action = "Expedite supply / release and protect ready orders"
        delay_text = "0" if o["material_delay_days"] == 0 else ("Unknown" if o["material_delay_days"] >= 120 else str(o["material_delay_days"]))
        rows.append({
            "Order": o["order"],
            "Remaining Qty": int(o["quantity"]),
            "Material Status": o["material_status"],
            "Expected Ready": expected_text,
            "Delay Days": delay_text,
            "Affected / Current Process": o["current_process"],
            "Action Needed": action,
        })
    return pd.DataFrame(rows)


def build_process_allocation(results):
    if results.empty:
        return pd.DataFrame()
    processes = process_options()
    rows = []
    amap, _, _, _, _, _ = machine_availability_map()
    for _, r in results.iterrows():
        row = {"Order": r["order"], "Current Process": r["current_process"]}
        for p in processes:
            row[p] = int(r["day1_workers"]) if r["current_process"] == p else 0
        row["Total Recommended Workers Today"] = int(r["day1_workers"])
        row["Process Machine Availability"] = f"{amap.get(r['current_process'], 1.0):.0%}"
        rows.append(row)
    return pd.DataFrame(rows)


def management_decision_rows(results, prepared, labour_factor, machine_table):
    rows = []
    p_lookup = {o["order"]: o for o in prepared}
    # Global labour issue only once.
    if labour_factor < 0.90:
        priority = "🔴 ACT NOW" if labour_factor < 0.75 else "🟠 ACTION TODAY"
        rows.append({
            "Priority": priority,
            "Issue": f"Labour availability {labour_factor:.0%}",
            "Affected": "All active orders",
            "Recommended Action": "Reallocate eligible/cross-trained workers to urgent ready orders; test approved overtime if schedule risk remains.",
            "When": "Today",
            "Expected Impact": "Protect scarce labour for the orders with the greatest due-date pressure.",
            "If No Action": "Capacity pressure can increase and more orders may move into projected delay.",
        })

    # Machine bottlenecks.
    if not machine_table.empty:
        for _, m in machine_table.sort_values("Availability %").head(3).iterrows():
            if float(m["Availability %"]) >= 90:
                continue
            affected_orders = [o["order"] for o in prepared if o["current_process"] == m["Process"]]
            priority = "🔴 ACT NOW" if float(m["Availability %"]) < 75 else "🟠 ACTION TODAY"
            rows.append({
                "Priority": priority,
                "Issue": f"{m['Process']} machine constraint ({float(m['Availability %']):.0f}%)",
                "Affected": ", ".join(affected_orders) if affected_orders else m["Process"],
                "Recommended Action": "Prioritize maintenance/recovery, confirm usable capacity, and protect urgent orders currently requiring this process.",
                "When": "Immediate" if priority.startswith("🔴") else "Today",
                "Expected Impact": "Reduce bottleneck pressure and improve effective production capacity at the constrained process.",
                "If No Action": "Orders entering this process can queue and projected completion dates may move later.",
            })

    for _, r in results.iterrows():
        o = p_lookup[r["order"]]
        if o["material_status"] in BLOCKED_MATERIAL_STATUSES:
            priority = "🔴 ACT NOW" if r["on_time"] == "NO" or str(r["risk"]).upper() == "HIGH" else "🟠 ACTION TODAY"
            rows.append({
                "Priority": priority,
                "Issue": f"Material {o['material_status']}",
                "Affected": f"Order {r['order']} · {o['current_process']}",
                "Recommended Action": "Expedite material clearance/replacement. Keep the blocked order out of eligible production allocation until material is released.",
                "When": "Immediate",
                "Expected Impact": "Restore the order's eligibility for production and prevent scarce resources being reserved for blocked work.",
                "If No Action": "The order remains blocked while its delivery buffer continues to reduce.",
            })
        elif o["material_status"] == "Partially Ready":
            rows.append({
                "Priority": "🟡 MONITOR",
                "Issue": "Material partially ready",
                "Affected": f"Order {r['order']}",
                "Recommended Action": "Confirm availability of components required for the next process before increasing allocation.",
                "When": "Before next stage",
                "Expected Impact": "Avoid starting work that cannot move smoothly to the next process.",
                "If No Action": "WIP may accumulate before the missing component/process stage.",
            })

        if int(r["projected_delay_days"]) > 0:
            rows.append({
                "Priority": "🔴 ACT NOW",
                "Issue": f"Projected delivery delay: {int(r['projected_delay_days'])} working day(s)",
                "Affected": f"Order {r['order']}",
                "Recommended Action": "Protect this order's eligible capacity and test machine recovery, workforce reallocation, approved overtime, subcontracting, or due-date recovery options.",
                "When": "Immediate",
                "Expected Impact": "Create the best available recovery path before the due-date gap becomes larger.",
                "If No Action": "The current simulation continues to project completion after the due date.",
            })
        elif str(r["risk"]).upper() == "HIGH":
            rows.append({
                "Priority": "🟠 ACTION TODAY",
                "Issue": "HIGH delivery risk",
                "Affected": f"Order {r['order']}",
                "Recommended Action": "Review the key risk drivers, protect the current process capacity, and monitor actual output against today's required production.",
                "When": "Today",
                "Expected Impact": "Reduce the chance that a high-risk order becomes a projected late order.",
                "If No Action": "A small disruption or output shortfall may move the order beyond its due-date buffer.",
            })
        elif str(r["risk"]).upper() == "MEDIUM":
            rows.append({
                "Priority": "🟡 MONITOR",
                "Issue": "MEDIUM delivery risk",
                "Affected": f"Order {r['order']}",
                "Recommended Action": "Track actual daily output and replan if material, labour or machine conditions worsen.",
                "When": "Daily review",
                "Expected Impact": "Provides early warning before corrective action becomes urgent.",
                "If No Action": "The order may lose its remaining schedule buffer without being noticed early.",
            })

    if not rows:
        rows.append({
            "Priority": "🟢 NO ACTION",
            "Issue": "No critical exception detected",
            "Affected": "Current operating plan",
            "Recommended Action": "Continue the current allocation and monitor daily production, materials, attendance and machine availability.",
            "When": "Daily review",
            "Expected Impact": "Maintain current delivery performance while preserving early-warning visibility.",
            "If No Action": "No immediate adverse effect is currently projected, but new disruptions can change the plan.",
        })

    priority_rank = {"🔴 ACT NOW": 0, "🟠 ACTION TODAY": 1, "🟡 MONITOR": 2, "🟢 NO ACTION": 3}
    out = pd.DataFrame(rows).drop_duplicates(subset=["Priority", "Issue", "Affected"])
    out["_rank"] = out["Priority"].map(priority_rank).fillna(9)
    return out.sort_values(["_rank", "Affected"]).drop(columns="_rank").reset_index(drop=True)


def current_snapshot(results, prepared, machine_table):
    orders_df = normalize_orders(st.session_state.orders)
    actual_workers = {
        str(r["Order"]): int(r["Workers Used Today"])
        for _, r in orders_df.iterrows()
        if str(r.get("Order", "")).strip() and int(r.get("Workers Used Today", 0)) > 0
    }
    actual_production = {
        str(r["Order"]): int(r["Actual Production Today"])
        for _, r in orders_df.iterrows()
        if str(r.get("Order", "")).strip() and int(r.get("Actual Production Today", 0)) > 0
    }
    return {
        "planning_date": str(st.session_state.planning_date),
        "workers_present": int(st.session_state.workers_present),
        "machines_available": int(machine_table["Available Today"].sum()) if not machine_table.empty else 0,
        "machine_constraints": int((machine_table["Availability %"] < 90).sum()) if not machine_table.empty else 0,
        "material_constraints": int(sum(o["material_status"] != "Ready" for o in prepared)),
        "high_risk_orders": int(sum(str(r).upper() == "HIGH" for r in results["risk"])) if not results.empty else 0,
        "projected_late": int((results["on_time"] == "NO").sum()) if not results.empty else 0,
        "projected_delay": int(results["projected_delay_days"].sum()) if not results.empty else 0,
        "recommended_workers": {str(r["order"]): int(r["day1_workers"]) for _, r in results.iterrows()},
        "actual_workers": actual_workers,
        "actual_production": actual_production,
    }


def change_monitor_df(history):
    if len(history) < 2:
        return pd.DataFrame()
    prev, cur = history[-2], history[-1]
    metrics = [
        ("Workers Present", "workers_present"),
        ("Machines Available", "machines_available"),
        ("Machine Constraints", "machine_constraints"),
        ("Material Constraints", "material_constraints"),
        ("HIGH-risk Orders", "high_risk_orders"),
        ("Projected Late Orders", "projected_late"),
        ("Projected Delay (Days)", "projected_delay"),
    ]
    rows = []
    for label, key in metrics:
        p, c = int(prev.get(key, 0)), int(cur.get(key, 0))
        delta = c - p
        delta_text = "—" if delta == 0 else f"{delta:+d}"
        rows.append({"Indicator": label, "Previous Plan": p, "Today": c, "Change": delta_text})
    return pd.DataFrame(rows)


def plot_daily_workforce(daily, history):
    if daily.empty:
        return None
    fig, ax = plt.subplots(figsize=(9.5, 3.7))
    orders = list(dict.fromkeys(daily["order"].tolist()))
    cmap = plt.get_cmap("tab10")
    for idx, order in enumerate(orders):
        colour = cmap(idx % 10)
        sub = daily[daily["order"] == order].copy()
        ax.plot(
            pd.to_datetime(sub["production_date"]),
            sub["workers_allocated"],
            linestyle="--",
            linewidth=2.0,
            color=colour,
            label=f"Order {order} · Planned",
        )
        actual_dates, actual_vals = [], []
        for snap in history:
            val = snap.get("actual_workers", {}).get(order)
            if val is not None:
                actual_dates.append(pd.Timestamp(snap["planning_date"]))
                actual_vals.append(val)
        if actual_dates:
            ax.plot(actual_dates, actual_vals, marker="o", linewidth=2.2, color=colour, label=f"Order {order} · Actual")
    ax.axvline(pd.Timestamp(st.session_state.planning_date), linewidth=1.1, alpha=.35)
    ax.set_ylabel("Workers Allocated")
    ax.set_xlabel("Production Date")
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.autofmt_xdate(rotation=0)
    ax.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    return fig


def choose_scenario_machine_process(machine_table):
    if machine_table.empty:
        return process_options()[0]
    # If everything is healthy, use the largest machine group (typically stitching) for a meaningful breakdown test.
    constrained = machine_table[machine_table["Availability %"] < 100]
    if not constrained.empty:
        return str(constrained.sort_values(["Availability %", "Total Machines"], ascending=[True, False]).iloc[0]["Process"])
    return str(machine_table.sort_values("Total Machines", ascending=False).iloc[0]["Process"])


def evaluate_scenarios(current_bundle, machine_table):
    prepared, _, _, results, _, _, _ = current_bundle
    baseline_on_time = int((results["on_time"] == "YES").sum()) if not results.empty else 0
    baseline_delay = int(results["projected_delay_days"].sum()) if not results.empty else 0
    baseline_risk = {str(r["order"]): str(r["risk"]).upper() for _, r in results.iterrows()}
    current_map, _, _, _, _, _ = machine_availability_map()
    target_process = choose_scenario_machine_process(machine_table)
    target_row = machine_table[machine_table["Process"] == target_process].iloc[0] if not machine_table.empty else None
    current_available = int(target_row["Available Today"]) if target_row is not None else 1
    total_target = int(target_row["Total Machines"]) if target_row is not None else 1
    breakdown_available = max(0, math.floor(current_available * 0.70))
    breakdown_map = dict(current_map)
    breakdown_map[target_process] = (breakdown_available / total_target) if total_target > 0 else 1.0

    active_order_ids = [o["order"] for o in prepared]
    material_target = None
    if active_order_ids:
        risk_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        material_target = sorted(
            prepared,
            key=lambda o: (-risk_rank.get(o["risk"], 0), o["days_remaining"]),
        )[0]["order"]

    workers_now = int(st.session_state.workers_present)
    moderate_workers = max(0, math.floor(workers_now * 0.85))
    severe_workers = max(0, math.floor(workers_now * 0.75))
    combined_map = dict(current_map)
    combined_map[target_process] = max(0.0, current_map.get(target_process, 1.0) * 0.80)
    severe_map = dict(current_map)
    severe_map[target_process] = max(0.0, current_map.get(target_process, 1.0) * 0.65)

    scenario_defs = [
        {
            "name": "Current Operating Plan",
            "assumption": "Today's actual labour, machine and material conditions",
            "workers": workers_now,
            "machine_map": current_map,
            "extra_delay": {},
            "status_override": {},
            "reason": "Baseline used to compare all hypothetical scenarios.",
            "response": "Continue monitoring and use Replan Today when actual conditions change.",
        },
        {
            "name": "Moderate Labour Shortage",
            "assumption": f"Workers present {workers_now} → {moderate_workers}",
            "workers": moderate_workers,
            "machine_map": current_map,
            "extra_delay": {},
            "status_override": {},
            "reason": "Reduced attendance lowers effective production capacity across active orders.",
            "response": "Reallocate eligible workers to urgent ready orders and test approved overtime if needed.",
        },
        {
            "name": "Critical Machine Breakdown",
            "assumption": f"{target_process}: {current_available} → {breakdown_available} machines available",
            "workers": workers_now,
            "machine_map": breakdown_map,
            "extra_delay": {},
            "status_override": {},
            "reason": f"Reduced {target_process} availability increases process bottleneck pressure.",
            "response": f"Prioritize maintenance/recovery at {target_process} and protect urgent work requiring that process.",
        },
        {
            "name": "Material Quality Hold",
            "assumption": f"Order {material_target}: Ready/Current → Quality Hold for 3 working days" if material_target else "Material hold on a priority order",
            "workers": workers_now,
            "machine_map": current_map,
            "extra_delay": {material_target: 3} if material_target else {},
            "status_override": {material_target: "Quality Hold"} if material_target else {},
            "reason": "The affected order temporarily loses production eligibility while its delivery buffer reduces.",
            "response": "Expedite inspection/replacement and reallocate eligible capacity to other ready orders until release.",
        },
        {
            "name": "Combined Resource Constraint",
            "assumption": f"Workers {workers_now} → {moderate_workers}; {target_process} -20%; priority material +2 days",
            "workers": moderate_workers,
            "machine_map": combined_map,
            "extra_delay": {material_target: 2} if material_target else {},
            "status_override": {material_target: "Awaiting Material"} if material_target else {},
            "reason": "Labour, machine and material constraints occur together and reduce schedule recovery flexibility.",
            "response": "Prioritize the most urgent ready orders, recover the bottleneck process, expedite material and test overtime/recovery actions.",
        },
        {
            "name": "Severe Combined Disruption",
            "assumption": f"Workers {workers_now} → {severe_workers}; {target_process} -35%; priority material +4 days",
            "workers": severe_workers,
            "machine_map": severe_map,
            "extra_delay": {material_target: 4} if material_target else {},
            "status_override": {material_target: "Rejected / Replacement Required"} if material_target else {},
            "reason": "Multiple severe constraints sharply reduce available capacity and due-date protection.",
            "response": "Activate a recovery plan: critical maintenance, material replacement, labour recovery, overtime/subcontracting and customer escalation where required.",
        },
    ]

    outputs = []
    configs = {}
    for sc in scenario_defs:
        bundle = calculate_plan(
            workers_override=sc["workers"],
            machine_map_override=sc["machine_map"],
            extra_delay_map=sc["extra_delay"],
            status_override_map=sc["status_override"],
        )
        sc_prepared, _, sc_ot, sc_res, _, _, _ = bundle
        on_time = int((sc_res["on_time"] == "YES").sum()) if not sc_res.empty else 0
        delay = int(sc_res["projected_delay_days"].sum()) if not sc_res.empty else 0
        high = int((sc_res["risk"].astype(str).str.upper() == "HIGH").sum()) if not sc_res.empty else 0
        late_orders = sc_res.loc[sc_res["on_time"] == "NO", "order"].astype(str).tolist() if not sc_res.empty else []
        changed_risk = [
            str(r["order"])
            for _, r in sc_res.iterrows()
            if baseline_risk.get(str(r["order"])) != str(r["risk"]).upper()
        ] if not sc_res.empty else []
        affected = list(dict.fromkeys(late_orders + changed_risk))
        if not affected and sc["name"] != "Current Operating Plan":
            affected = [o["order"] for o in sc_prepared if o["machine_availability"] < 0.90 or o["material_status"] != "Ready"]
        delay_delta = delay - baseline_delay
        ontime_delta = on_time - baseline_on_time
        if sc["name"] == "Current Operating Plan":
            impact = "🟢 STABLE" if delay == 0 else ("🟠 WARNING" if delay <= 3 else "🔴 HIGH IMPACT")
            comparison = "● BASELINE"
        else:
            if ontime_delta >= 0 and delay_delta <= 0:
                impact = "🟢 STABLE"
                comparison = "● NO MAJOR CHANGE" if ontime_delta == 0 and delay_delta == 0 else "▲ BETTER"
            elif ontime_delta == 0 and delay_delta <= 2:
                impact = "🟡 WATCH"
                comparison = "▼ WORSE"
            elif ontime_delta >= -1 and delay_delta <= 5:
                impact = "🟠 WARNING"
                comparison = "▼ WORSE"
            elif ontime_delta >= -2 and delay_delta <= 10:
                impact = "🔴 HIGH IMPACT"
                comparison = "▼▼ MUCH WORSE"
            else:
                impact = "🔴 CRITICAL"
                comparison = "▼▼ MUCH WORSE"
        outputs.append({
            "Scenario": sc["name"],
            "Impact": impact,
            "Comparison": comparison,
            "Assumption Tested": sc["assumption"],
            "Affected Orders": ", ".join(affected) if affected else "No additional order exception",
            "Orders On Time": f"{on_time}/{len(sc_res)}",
            "HIGH-risk Orders": high,
            "Projected Delay (Days)": delay,
            "Recommended Overtime": f"{sc_ot:.1%}",
            "Why It Matters": sc["reason"],
            "Suggested Response": sc["response"],
        })
        configs[sc["name"]] = {**sc, "bundle": bundle}
    return pd.DataFrame(outputs), configs


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sp-brand">
          <div class="sp-brand-icon">⬢</div>
          <div><div class="sp-brand-title">SURE-Plan</div><div class="sp-brand-sub">SMART DECISIONS. RELIABLE DELIVERY.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_options = ["Dashboard", "Reports"] + (["Admin Settings"] if role == "Admin" else [])
    page = st.radio("Navigation", nav_options, label_visibility="collapsed")
    st.divider()
    st.caption("SIGNED IN AS")
    st.write(f"**{st.session_state.get('username','planner')}** · {role}")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("↪ Sign Out", use_container_width=True):
        logout()
    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    st.caption("SURE-Plan v3.0")
    st.caption("Explainable AI + Rolling Operations Planning")

# -----------------------------------------------------------------------------
# Admin Settings
# -----------------------------------------------------------------------------
if page == "Admin Settings":
    st.markdown(
        "<div class='top-head'><h1>Admin Settings</h1><p>Configure factory reference values and the process-wise machine master used by the daily planning model.</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("<div class='section-title'>Factory Reference Settings</div>", unsafe_allow_html=True)
        settings["company_name"] = st.text_input("Company / model label", settings["company_name"])
        settings["benchmark_workers"] = st.number_input(
            "Standard Production Workforce",
            min_value=1,
            max_value=1000,
            value=int(settings["benchmark_workers"]),
            step=1,
            help="Reference production workforce. Planner enters actual attendance separately each day.",
        )
        settings["base_capacity"] = st.number_input(
            "Standard Production Capacity (uppers/day)",
            min_value=10.0,
            max_value=10000.0,
            value=float(settings["base_capacity"]),
            step=10.0,
            help="Editable production-capacity benchmark. Replace with verified observed company capacity when available.",
        )
        settings["max_overtime"] = st.slider(
            "Maximum Overtime Capacity",
            min_value=0.00,
            max_value=0.30,
            value=float(settings["max_overtime"]),
            step=0.01,
        )
        st.warning("Standard workforce, machine counts and production capacity are prototype/reference assumptions until verified with company records.")
    with c2:
        st.markdown("<div class='section-title'>Current Reference Snapshot</div>", unsafe_allow_html=True)
        total_machine_ref = int(clean_machine_master(st.session_state.machine_master)["Total Machines"].sum())
        ref_df = pd.DataFrame(
            [
                ["Standard workforce", int(settings["benchmark_workers"]), "workers"],
                ["Standard production capacity", int(settings["base_capacity"]), "uppers/day"],
                ["Machine master total", total_machine_ref, "machines"],
                ["Maximum overtime", f"{settings['max_overtime']:.0%}", "capacity buffer"],
            ],
            columns=["Reference", "Value", "Unit"],
        )
        st.dataframe(ref_df, use_container_width=True, hide_index=True, height=180)
        st.info("Changing the standard workforce does not automatically change production capacity. Capacity remains an independently editable benchmark.")

    st.markdown("<div class='section-title'>Process-wise Machine Master</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Maintain the production-machine reference from material cutting through final shoe-upper preparation. The total is auto-calculated from these rows.</div>", unsafe_allow_html=True)
    edited_master = st.data_editor(
        clean_machine_master(st.session_state.machine_master),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Sequence": st.column_config.NumberColumn("Seq.", min_value=1, step=1, required=True, width="small"),
            "Process": st.column_config.TextColumn("Production Process", required=True, width="medium"),
            "Machine Type": st.column_config.TextColumn("Machine Type", required=True, width="large"),
            "Total Machines": st.column_config.NumberColumn("Total Machines", min_value=0, step=1, required=True, width="small"),
        },
        key="admin_machine_master_editor",
    )
    st.session_state.machine_master = clean_machine_master(edited_master)
    sync_machine_today()
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("Total Machine Types / Processes", len(st.session_state.machine_master))
    with mc2:
        st.metric("Total Machines", int(st.session_state.machine_master["Total Machines"].sum()))
    with mc3:
        st.metric("Standard Workforce", int(settings["benchmark_workers"]))

    with st.expander("Explainable Decision Tree & Academic Model Note"):
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
        st.caption("Prototype Model Confidence is the classifier's estimated class probability from researcher-designed simulated training data; it is not certainty of the actual delivery outcome.")
        st.caption("Machine-master values shown by default are estimated prototype benchmarks and should be replaced with verified factory counts before claiming company-specific results.")
    st.stop()

# Synchronize today's machine input with the current Admin master.
sync_machine_today()

# -----------------------------------------------------------------------------
# Reports page
# -----------------------------------------------------------------------------
if page == "Reports":
    bundle = calculate_plan()
    prepared, pred_df, overtime, results, daily, resource_info, recommendations = bundle
    machine_table = machine_status_table()
    decision_df = management_decision_rows(results, prepared, resource_info.get("labour_availability", 0), machine_table) if not results.empty else pd.DataFrame()
    material_df = build_material_tracker(prepared)
    st.markdown(
        "<div class='top-head'><h1>Reports & Export</h1><p>Download the current rolling plan, machine status, material status and management-decision outputs.</p></div>",
        unsafe_allow_html=True,
    )
    if results.empty:
        st.info("Add valid active orders on the Dashboard first.")
        st.stop()

    export_results = results.copy()
    export_results["projected_completion_date"] = export_results["completion_day"].apply(
        lambda x: add_business_days(st.session_state.planning_date, int(x) - 1).date()
    )
    export_results = export_results.rename(columns={"projected_delay_days": "projected_delay_days_working"})

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.download_button("⬇ Order Results CSV", export_results.to_csv(index=False).encode("utf-8"), "SURE_Plan_Order_Results.csv", "text/csv", use_container_width=True)
    with d2:
        st.download_button("⬇ Daily Allocation CSV", daily.to_csv(index=False).encode("utf-8"), "SURE_Plan_Daily_Workforce_Allocation.csv", "text/csv", use_container_width=True)
    with d3:
        st.download_button("⬇ Machine Status CSV", machine_table.to_csv(index=False).encode("utf-8"), "SURE_Plan_Machine_Status.csv", "text/csv", use_container_width=True)
    with d4:
        st.download_button("⬇ Decision Centre CSV", decision_df.to_csv(index=False).encode("utf-8"), "SURE_Plan_Management_Decisions.csv", "text/csv", use_container_width=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button("⬇ Material Tracker CSV", material_df.to_csv(index=False).encode("utf-8"), "SURE_Plan_Material_Tracker.csv", "text/csv", use_container_width=True)
    with e2:
        history_df = pd.DataFrame(st.session_state.plan_history)
        st.download_button("⬇ Replan History CSV", history_df.to_csv(index=False).encode("utf-8"), "SURE_Plan_Replan_History.csv", "text/csv", use_container_width=True)
    with e3:
        amap, bottleneck_factor, bottleneck_process, total_m, avail_m, overall_m = machine_availability_map()
        summary_text = f"""SURE-PLAN MANAGEMENT SUMMARY\n\nPlanning date: {st.session_state.planning_date}\nWorkers present today: {st.session_state.workers_present}\nStandard workforce: {settings['benchmark_workers']}\nLabour availability factor: {resource_info.get('labour_availability',0):.1%}\nMachines available: {avail_m}/{total_m} ({overall_m:.1%})\nCurrent bottleneck process: {bottleneck_process} ({bottleneck_factor:.1%} available)\nStandard capacity benchmark: {settings['base_capacity']:.0f} uppers/day\nRecommended overtime: {overtime:.1%}\n\nOrder results:\n{export_results.to_string(index=False)}\n\nManagement Decision Centre:\n{decision_df.to_string(index=False)}\n\nAcademic note:\nThe ML model is a proof-of-concept trained on researcher-designed simulated scenarios.\nMachine counts and capacity values are prototype/reference assumptions unless independently verified with company records.\n"""
        st.download_button("⬇ Management Summary TXT", summary_text.encode("utf-8"), "SURE_Plan_Management_Summary.txt", "text/plain", use_container_width=True)

    st.markdown("<div class='section-title'>Order Results</div>", unsafe_allow_html=True)
    st.dataframe(export_results, use_container_width=True, hide_index=True)
    st.markdown("<div class='section-title'>Machine Availability by Process</div>", unsafe_allow_html=True)
    st.dataframe(machine_table, use_container_width=True, hide_index=True)
    st.markdown("<div class='section-title'>Management Decision Centre</div>", unsafe_allow_html=True)
    st.dataframe(decision_df, use_container_width=True, hide_index=True)
    st.stop()

# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="top-head">
      <h1>Production Decision Dashboard</h1>
      <p>Daily production planning, workforce allocation, process-wise machine availability, material readiness & delivery-risk analysis · Signed in as {role}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 1. Daily operating inputs
st.markdown("<div class='section-title'>1. Today's Production Inputs</div>", unsafe_allow_html=True)
with st.container(border=True):
    t1, t2, t3, t4 = st.columns([1.0, 1.0, 1.0, 1.0])
    with t1:
        st.session_state.planning_date = st.date_input("Planning Date", value=st.session_state.planning_date)
    with t2:
        max_workers_input = max(500, int(settings["benchmark_workers"]) * 2)
        st.session_state.workers_present = st.number_input(
            "Workers Present Today",
            min_value=0,
            max_value=max_workers_input,
            value=int(st.session_state.workers_present),
            step=1,
            help="Enter actual production attendance for today. The ML labour factor is capped at the configured standard workforce because the prototype training data does not model >100% labour availability.",
        )
    with t3:
        st.metric("Standard Workforce", int(settings["benchmark_workers"]), help="Admin reference; editable only in Admin Settings.")
    with t4:
        st.metric("Standard Capacity", f"{settings['base_capacity']:.0f} /day", help="Independent Admin capacity benchmark; it does not auto-scale when workforce is changed.")

    st.markdown("**Active Orders & Daily Progress**")
    st.caption("Enter actual production progress and the order's current production process. Remaining quantity is calculated automatically and is used for risk/replanning.")
    st.session_state.orders = normalize_orders(st.session_state.orders)
    edited_orders = st.data_editor(
        st.session_state.orders,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Order": st.column_config.TextColumn("Order ID", required=True, width="small"),
            "Original Quantity": st.column_config.NumberColumn("Original Qty", min_value=1, step=50, required=True, width="small"),
            "Completed Before Today": st.column_config.NumberColumn("Completed Before Today", min_value=0, step=10, width="small"),
            "Actual Production Today": st.column_config.NumberColumn("Actual Production Today", min_value=0, step=10, width="small", help="Today's completed output for this order."),
            "Workers Used Today": st.column_config.NumberColumn("Actual Workers Today", min_value=0, step=1, width="small", help="Optional actual worker allocation. Recording this enables Actual vs Planned workforce history."),
            "Due Date": st.column_config.DateColumn("Due Date", required=True, width="small"),
            "Material Status": st.column_config.SelectboxColumn("Material Readiness", options=MATERIAL_STATUSES, required=True, width="medium"),
            "Expected Material Ready Date": st.column_config.DateColumn("Expected Material Ready", width="small", help="Required operationally when material is unavailable/held/short/rejected."),
            "Current Process": st.column_config.SelectboxColumn("Current Production Process", options=process_options(), required=True, width="medium"),
        },
        key="active_orders_editor",
    )
    st.session_state.orders = normalize_orders(edited_orders)

    remaining_view = st.session_state.orders[["Order", "Original Quantity", "Completed Before Today", "Actual Production Today"]].copy()
    remaining_view["Remaining Quantity"] = (
        remaining_view["Original Quantity"] - remaining_view["Completed Before Today"] - remaining_view["Actual Production Today"]
    ).clip(lower=0)
    with st.expander("View calculated remaining quantities"):
        st.dataframe(remaining_view, use_container_width=True, hide_index=True)

# Machine availability input
st.markdown("<div class='section-title'>2. Process-wise Machine Availability Today</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Mark how many machines are available at every production stage and record breakdown/maintenance issues. SURE-Plan uses the current process availability of each order in risk and planning calculations.</div>", unsafe_allow_html=True)
with st.container(border=True):
    mt = st.session_state.machine_today.copy()
    edited_mt = st.data_editor(
        mt,
        use_container_width=True,
        hide_index=True,
        disabled=["Sequence", "Process", "Machine Type", "Total Machines"],
        column_config={
            "Sequence": st.column_config.NumberColumn("Seq.", width="small"),
            "Process": st.column_config.TextColumn("Production Process", width="medium"),
            "Machine Type": st.column_config.TextColumn("Machine Type", width="medium"),
            "Total Machines": st.column_config.NumberColumn("Total", width="small"),
            "Available Today": st.column_config.NumberColumn("Available Today", min_value=0, step=1, width="small"),
            "Breakdown / Issue": st.column_config.TextColumn("Breakdown / Maintenance / Issue", width="large"),
        },
        key="machine_today_editor",
    )
    # Sanitize any value entered above the Admin total.
    for idx in edited_mt.index:
        total = int(edited_mt.at[idx, "Total Machines"])
        raw_available = pd.to_numeric(edited_mt.at[idx, "Available Today"], errors="coerce")
        available_value = 0 if pd.isna(raw_available) else int(raw_available)
        edited_mt.at[idx, "Available Today"] = max(0, min(total, available_value))
    st.session_state.machine_today = edited_mt
    machine_table = machine_status_table()
    amap, bottleneck_factor, bottleneck_process, total_machines, available_machines, overall_machine_pct = machine_availability_map()
    cma, cmb, cmc, cmd = st.columns(4)
    with cma:
        st.metric("Machines Available Today", f"{available_machines}/{total_machines}")
    with cmb:
        st.metric("Overall Machine Availability", f"{overall_machine_pct:.1%}")
    with cmc:
        st.metric("Bottleneck Process", bottleneck_process)
    with cmd:
        st.metric("Bottleneck Availability", f"{bottleneck_factor:.1%}")
    st.dataframe(
        machine_table[["Process", "Total Machines", "Available Today", "Breakdown / Unavailable", "Availability %", "Status", "Breakdown / Issue"]],
        use_container_width=True,
        hide_index=True,
        height=300,
    )
    st.caption("Availability is not the same as true machine utilization. True utilization requires process run-hours/cycle-time data; this prototype does not invent those values.")

replan_clicked = st.button("▶ REPLAN TODAY", type="primary", use_container_width=True)

bundle = calculate_plan()
prepared, pred_df, overtime, results, daily, resource_info, recommendations = bundle

if results.empty:
    st.info("Enter at least one active order with remaining quantity to generate the management dashboard.")
    st.stop()

# Save a daily plan snapshot only when the planner explicitly replans.
if replan_clicked:
    snap = current_snapshot(results, prepared, machine_table)
    st.session_state.plan_history.append(snap)
    st.session_state.plan_history = st.session_state.plan_history[-90:]
    st.toast("Today's conditions were analysed and the rolling plan was updated.")

# Validate blocked material rows without a ready date.
missing_ready = [
    o["order"] for o in prepared
    if o["material_status"] in BLOCKED_MATERIAL_STATUSES and pd.isna(o.get("expected_material_ready"))
]
if missing_ready:
    st.warning(f"Expected Material Ready Date is missing for blocked order(s): {', '.join(missing_ready)}. They are kept out of production allocation until a ready date is entered.")

# 3. KPI summary
risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
worst_risk = max((str(r).upper() for r in results["risk"]), key=lambda x: risk_rank.get(x, 0))
late_count = int((results["on_time"] == "NO").sum())
if late_count > 0 and worst_risk == "LOW":
    worst_risk = "MEDIUM"
risk_class = "risk-high" if worst_risk == "HIGH" else ("risk-medium" if worst_risk == "MEDIUM" else "risk-low")
labour_factor = resource_info.get("labour_availability", 0.0)
effective_capacity = resource_info.get("effective_daily_capacity", 0.0)
total_delay = int(results["projected_delay_days"].sum())
on_time = int((results["on_time"] == "YES").sum())
active_orders = len(results)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"<div class='card {risk_class}'><div class='card-label'>Overall Delivery Risk</div><div class='card-value'>{worst_risk}</div><div class='card-sub'>{late_count} order(s) currently projected late</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='card'><div class='card-label'>Available Capacity Today</div><div class='card-value'>{effective_capacity:.0f}</div><div class='card-sub'>uppers/day using labour, bottleneck machine factor & overtime</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='card'><div class='card-label'>Workers Present</div><div class='card-value'>{int(st.session_state.workers_present)}</div><div class='card-sub'>{labour_factor:.1%} of standard workforce used by model</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='card'><div class='card-label'>Machines Available</div><div class='card-value'>{available_machines}/{total_machines}</div><div class='card-sub'>{bottleneck_process}: {bottleneck_factor:.0%} availability</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='card'><div class='card-label'>Orders On Time</div><div class='card-value'>{on_time}/{active_orders}</div><div class='card-sub'>under the current rolling production plan</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='card'><div class='card-label'>Projected Delay</div><div class='card-value'>{total_delay}</div><div class='card-sub'>total working days across active orders</div></div>", unsafe_allow_html=True)

st.markdown(
    f"<div class='pill-row'><span class='pill'>Labour availability: {labour_factor:.1%}</span><span class='pill'>Machine availability: {overall_machine_pct:.1%}</span><span class='pill'>Bottleneck factor: {bottleneck_factor:.1%}</span><span class='pill'>Recommended overtime: {overtime:.1%}</span><span class='pill'>Material constraints: {sum(o['material_status'] != 'Ready' for o in prepared)}</span></div>",
    unsafe_allow_html=True,
)

# 4. Risk + resource allocation
left, right = st.columns([1.12, 1])
with left:
    st.markdown("<div class='section-title'>3. AI/ML Delivery-Risk Assessment</div>", unsafe_allow_html=True)
    st.dataframe(pred_df, use_container_width=True, hide_index=True, height=260)
    st.caption("LOW = manageable/on track · MEDIUM = warning/preventive attention · HIGH = serious delivery risk. Risk uses remaining quantity, time, labour, current-process machine availability, material delay, required output and capacity ratio.")
    st.caption("Prototype Model Confidence is the decision-tree class probability from simulated training data; it is not certainty of the actual delivery outcome.")

with right:
    st.markdown("<div class='section-title'>4. Resource Allocation Summary</div>", unsafe_allow_html=True)
    alloc = results[["order", "day1_workers", "current_process", "completion_day", "projected_delay_days", "on_time"]].copy()
    alloc["Projected Completion Date"] = alloc["completion_day"].apply(lambda x: add_business_days(st.session_state.planning_date, int(x) - 1).strftime("%d %b %Y"))
    alloc = alloc[["order", "day1_workers", "current_process", "Projected Completion Date", "projected_delay_days", "on_time"]]
    alloc.columns = ["Order", "Recommended Workers Today", "Primary Process Today", "Projected Completion", "Projected Delay (Days)", "On Time?"]
    st.dataframe(alloc, use_container_width=True, hide_index=True, height=260)
    st.caption("Recommended Workers Today is the share of today's available workforce assigned to each active order for the current planning period—not the total labour required for the entire order lifecycle.")
    with st.expander("View process-wise worker allocation"):
        process_alloc = build_process_allocation(results)
        st.dataframe(process_alloc, use_container_width=True, hide_index=True)
        st.caption("To avoid false precision, this prototype assigns each order's recommended workers to the planner-selected current production process. It does not allocate the same workers across every stage simultaneously. Multi-stage WIP/skill routing can be added when real shop-floor routing data are available.")

# 5. Machine + material visibility
mleft, mright = st.columns([1.02, 1])
with mleft:
    st.markdown("<div class='section-title'>5. Machine Availability & Bottleneck View</div>", unsafe_allow_html=True)
    machine_display = machine_table[["Process", "Total Machines", "Available Today", "Breakdown / Unavailable", "Availability %", "Status"]].copy()
    machine_display["Availability %"] = machine_display["Availability %"].round(1)
    st.dataframe(machine_display, use_container_width=True, hide_index=True, height=300)
    st.caption(f"Current bottleneck: {bottleneck_process} ({bottleneck_factor:.1%} available). Breakdown notes entered above are used in the management action view.")
with mright:
    st.markdown("<div class='section-title'>6. Material Readiness & Constraint Tracker</div>", unsafe_allow_html=True)
    material_df = build_material_tracker(prepared)
    st.dataframe(material_df, use_container_width=True, hide_index=True, height=300)
    st.caption("Orders on Awaiting Material, Quality Hold, Shortage or Rejected/Replacement status are blocked from eligible production allocation until their material-ready condition is restored.")

# 7. Daily change monitor
st.markdown("<div class='section-title'>7. Daily Change Monitor</div>", unsafe_allow_html=True)
change_df = change_monitor_df(st.session_state.plan_history)
if change_df.empty:
    st.info("Run REPLAN TODAY on at least two planning updates to compare the previous plan with today's conditions. Actual Production Today and Actual Workers Today are recorded in the replan history when entered.")
else:
    st.dataframe(change_df, use_container_width=True, hide_index=True, height=285)

# 8. Workforce allocation + management decision centre
left2, right2 = st.columns([1.03, 1])
with left2:
    st.markdown("<div class='section-title'>8. Daily Workforce Allocation by Order</div>", unsafe_allow_html=True)
    fig = plot_daily_workforce(daily, st.session_state.plan_history)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.caption("X-axis = Production Date · Y-axis = Workers Allocated. Solid lines show recorded actual worker use when entered and saved through Replan Today; dashed lines show SURE-Plan's future planned allocation.")

with right2:
    st.markdown("<div class='section-title'>9. Management Decision Centre</div>", unsafe_allow_html=True)
    decision_df = management_decision_rows(results, prepared, labour_factor, machine_table)
    act_now = int((decision_df["Priority"] == "🔴 ACT NOW").sum())
    action_today = int((decision_df["Priority"] == "🟠 ACTION TODAY").sum())
    if act_now:
        st.markdown(f"<div class='alert-bad'>⚠ ACTION REQUIRED · {act_now} critical management action(s) identified.</div>", unsafe_allow_html=True)
    elif action_today:
        st.markdown(f"<div class='alert-warn'>Attention needed today · {action_today} corrective action(s) identified.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='alert-good'>✓ No immediate corrective action is required under the current plan.</div>", unsafe_allow_html=True)
    st.dataframe(decision_df[["Priority", "Issue", "Affected", "Recommended Action", "When"]].head(8), use_container_width=True, hide_index=True, height=305)
    with st.expander("Why, expected impact & consequence if no action"):
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
    st.caption("Management Priority is separate from ML Delivery Risk: ACT NOW/ACTION TODAY/MONITOR/NO ACTION describes how urgently management should respond. Use What-if Impact Analysis → Test Recovery Action below to test selected interventions before changing the active plan.")

# 10. What-if impact analysis
st.markdown("<div class='section-title'>10. What-if Impact Analysis</div>", unsafe_allow_html=True)
st.markdown("<div class='section-sub'>Hypothetical simulations only. They compare possible labour, machine and material disruptions against today's Current Operating Plan and do not modify the active production plan.</div>", unsafe_allow_html=True)
scenario_df, scenario_configs = evaluate_scenarios(bundle, machine_table)
scenario_cards = scenario_df.head(6).to_dict("records")
card_cols = st.columns(len(scenario_cards))
for col, sc in zip(card_cols, scenario_cards):
    with col:
        badge_colour = "#139b66" if "STABLE" in sc["Impact"] else ("#d58a08" if "WATCH" in sc["Impact"] or "WARNING" in sc["Impact"] else "#e04466")
        st.markdown(
            f"<div class='scenario'><div class='scenario-title'>{sc['Scenario']}</div><div class='scenario-badge' style='color:{badge_colour}'>{sc['Impact']}</div><div class='scenario-num'>{sc['Orders On Time']}</div><div class='scenario-sub'>Orders on time · {sc['Projected Delay (Days)']} projected delay day(s)<br>{sc['Comparison']}</div></div>",
            unsafe_allow_html=True,
        )

with st.expander("View what-if assumptions, affected orders, reasons and suggested responses"):
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

with st.expander("Test Recovery Action"):
    non_baseline = [s for s in scenario_df["Scenario"].tolist() if s != "Current Operating Plan"]
    rc1, rc2 = st.columns(2)
    with rc1:
        selected_scenario = st.selectbox("Scenario to recover", non_baseline)
    with rc2:
        recovery_action = st.selectbox(
            "Recovery action to test",
            [
                "Use Maximum Overtime",
                "Recover 50% of Machine Loss",
                "Recover 10% of Standard Workforce",
                "Expedite Material by 2 Working Days",
            ],
        )
    if st.button("Test Recovery Action", use_container_width=True):
        cfg = copy.deepcopy(scenario_configs[selected_scenario])
        before_bundle = cfg["bundle"]
        workers = int(cfg["workers"])
        m_map = dict(cfg["machine_map"])
        extra_delay = dict(cfg["extra_delay"])
        status_override = dict(cfg["status_override"])
        overtime_override = None

        if recovery_action == "Use Maximum Overtime":
            overtime_override = float(settings["max_overtime"])
        elif recovery_action == "Recover 50% of Machine Loss":
            current_map, _, _, _, _, _ = machine_availability_map()
            for p in m_map:
                loss = current_map.get(p, 1.0) - m_map[p]
                if loss > 0:
                    m_map[p] = min(current_map.get(p, 1.0), m_map[p] + loss * 0.50)
        elif recovery_action == "Recover 10% of Standard Workforce":
            workers = min(int(st.session_state.workers_present), workers + max(1, round(int(settings["benchmark_workers"]) * 0.10)))
        elif recovery_action == "Expedite Material by 2 Working Days":
            extra_delay = {k: max(0, int(v) - 2) for k, v in extra_delay.items()}
            if all(v == 0 for v in extra_delay.values()):
                status_override = {}

        after_bundle = calculate_plan(
            workers_override=workers,
            machine_map_override=m_map,
            extra_delay_map=extra_delay,
            status_override_map=status_override,
            overtime_override=overtime_override,
        )
        before_res = before_bundle[3]
        after_res = after_bundle[3]
        before_on = int((before_res["on_time"] == "YES").sum())
        after_on = int((after_res["on_time"] == "YES").sum())
        before_delay = int(before_res["projected_delay_days"].sum())
        after_delay = int(after_res["projected_delay_days"].sum())
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.metric("Before: Orders On Time", f"{before_on}/{len(before_res)}")
        with r2:
            st.metric("After: Orders On Time", f"{after_on}/{len(after_res)}", delta=after_on - before_on)
        with r3:
            st.metric("Before: Projected Delay", before_delay)
        with r4:
            st.metric("After: Projected Delay", after_delay, delta=after_delay - before_delay, delta_color="inverse")
        st.caption("Recovery-test results are recalculated by the prototype planning engine. They are decision-support estimates, not guaranteed production outcomes.")

with st.expander("Management detail: order-level reasons, process allocation & full rolling schedule"):
    st.markdown("**Order-level recommendations**")
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.markdown("**Process-wise worker allocation today**")
    st.dataframe(build_process_allocation(results), use_container_width=True, hide_index=True)
    st.markdown("**Full planned daily allocation**")
    st.dataframe(daily, use_container_width=True, hide_index=True, height=360)

st.markdown(
    "<div class='footer-line'>SURE-Plan MBA prototype · Rolling daily production planning + explainable ML + disruption-aware resource reallocation. Academic disclosure: ML training scenarios are researcher-designed/simulated; default workforce, machine counts and capacity values are prototype/reference assumptions unless verified with company records. Session-state history is prototype persistence and may reset when the Streamlit app session/cloud instance restarts.</div>",
    unsafe_allow_html=True,
)
