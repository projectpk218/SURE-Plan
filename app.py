from pathlib import Path
from datetime import date
import copy
import math
import os
import json
import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from rapid_storage import RapidStore, StorageConflict, StorageConfigurationError, configuration_payload, operations_payload, frame_payload, payload_frame
from rapid_workspace import Workspace
from production_management import render_production_management, daily_progress, action_frame, action_summary

from planner_core import (
    business_days_between,
    business_day_deadline,
    validate_order_ids,
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
    page_title="RAPID | Production Decision Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# RAPID management-dashboard styling.
# The calculation/ML logic is unchanged; this block only controls presentation.
# -----------------------------------------------------------------------------
st.markdown(f"<style>{(BASE / 'rapid_theme.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


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
            <div style="background:#FFFEFB;border:1px solid #DFD9CC;border-radius:18px;padding:28px 30px;box-shadow:0 12px 35px rgba(16,60,74,.08)">
              <div style="font-size:30px;font-weight:850;color:#103C4A">⬢ RAPID</div>
              <div style="color:#61747A;margin-top:5px">Production Decision Dashboard</div>
              <div style="font-size:12px;color:#61747A;margin-top:6px">Explainable AI + rolling operations planning</div>
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


database_url = os.environ.get("RAPID_DATABASE_URL") or get_secret("RAPID_DATABASE_URL", None)
if database_url and not database_url.startswith("sqlite:"):
    if not all(get_secret(key, None) for key in ("ADMIN_USERNAME", "ADMIN_PASSWORD", "USER_USERNAME", "USER_PASSWORD")):
        st.error("Configure Admin and Planner login credentials in Streamlit Secrets before using the shared database.")
        st.stop()

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
    current_status = str(row.get("Material Status", "Ready"))
    status = status_override or current_status
    expected = row.get("Expected Material Ready Date")
    planning_delay = 0
    # Scenario duration extends the real material delay. A hypothetical hold
    # must not create an unknown 120-day base delay for an otherwise ready order.
    if current_status in BLOCKED_MATERIAL_STATUSES:
        if pd.isna(expected):
            # Keep the order effectively blocked in the operations simulation until a ready date is entered.
            planning_delay = 120
        else:
            planning_delay = max(1, business_days_until_ready(planning_date_value, expected))
    planning_delay += max(0, int(extra_delay))
    # Training scenarios only cover 0-6 material-delay days. Keep ML input in-range and disclose it.
    ml_delay = min(6, planning_delay)
    eligible_today = status not in BLOCKED_MATERIAL_STATUSES and planning_delay == 0
    if status == "Partially Ready" and planning_delay == 0:
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
    try:
        validate_order_ids(orders_df.loc[orders_df["Order"] != "", "Order"])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
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
            "due_day": business_day_deadline(st.session_state.planning_date, row["Due Date"]),
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

        if pd.isna(r["completion_day"]):
            rows.append({
                "Priority": "🔴 ACT NOW",
                "Issue": "No completion within planning horizon",
                "Affected": f"Order {r['order']}",
                "Recommended Action": "Resolve resource/material constraints and replan before committing a completion date.",
                "When": "Immediate",
                "Expected Impact": "Establish a feasible completion forecast; reported projected delay is a minimum only.",
                "If No Action": "Completion remains unknown under the current assumptions.",
            })
        elif int(r["projected_delay_days"]) > 0:
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
    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.patch.set_facecolor("#FFFEFB")
    ax.set_facecolor("#FFFEFB")
    orders = list(dict.fromkeys(daily["order"].tolist()))
    chart_colours = ["#087E8B", "#B38A4A", "#375F85", "#547A65", "#9C5655", "#70678B"]
    for idx, order in enumerate(orders):
        colour = chart_colours[idx % len(chart_colours)]
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
# UI helpers — presentation only
# -----------------------------------------------------------------------------
def ui_kpi(icon, label, value, sub, tone="blue", value_class=""):
    st.markdown(
        f"""
        <div class="kpi-card kpi-{tone}">
          <div class="kpi-top"><div class="kpi-icon">{icon}</div><div class="kpi-label">{label}</div></div>
          <div class="kpi-value {value_class}">{value}</div>
          <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_section(title, icon="", subtitle=None):
    st.markdown(f"<div class='section-title'><span class='section-icon'>{icon}</span>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)


def ui_header(title, subtitle):
    planning_text = pd.Timestamp(st.session_state.planning_date).strftime("%d %b %Y") if "planning_date" in st.session_state else "—"
    role_text = st.session_state.get("role", "Planner")
    initial = "A" if role_text == "Admin" else "P"
    st.markdown(
        f"""
        <div class="dashboard-header">
          <div class="dashboard-title-wrap">
            <div class="menu-orb">☰</div>
            <div><div class="dashboard-h1">{title}</div><div class="dashboard-sub">{subtitle}</div></div>
          </div>
          <div class="header-right">
            <div class="header-chip"><div class="header-chip-k">Planning Date</div><div class="header-chip-v">▣ &nbsp; {planning_text}</div></div>
            <div class="header-chip avatar-chip"><div class="avatar-circle">{initial}</div><div><div class="header-chip-v">{role_text}</div><div class="header-chip-k">{st.session_state.get('username','planner')}</div></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Shared persistence is separate from all production and ML calculations.
@st.cache_resource
def get_database(url):
    return RapidStore(url)


def show_storage_error(exc):
    if isinstance(exc, (StorageConflict, ValueError, PermissionError, StorageConfigurationError)):
        st.error(str(exc))
    else:
        st.error("The database could not complete this operation. Your draft is still in this session. Check the connection and retry; no successful save has been confirmed.")


try:
    if not database_url:
        st.error("Permanent storage is not configured. Add RAPID_DATABASE_URL in Streamlit Secrets, then reload. See STORAGE_SETUP.md for the setup steps.")
        st.stop()
    database = get_database(database_url)
    sync_machine_today()
    database.initialize(configuration_payload(st.session_state), operations_payload(st.session_state))
    workspace = Workspace(database, st.session_state, st.session_state.get("username", ""), role)
    if not st.session_state.get("_storage_loaded"):
        workspace.reload()
    settings = st.session_state.settings
    widget_epoch = st.session_state.get("_widget_epoch", 0)
except Exception as exc:
    show_storage_error(exc)
    st.stop()


def show_saved_records():
    with st.expander("Saved daily records & data backup"):
        try:
            dates = database.daily_dates()
            if dates:
                selected_date = st.selectbox("Saved production date", dates, key="saved_record_date")
                saved = database.daily_record(selected_date)
                st.caption("Read-only record: inputs, reference settings and outputs exactly as saved. Earlier records are never recalculated using today's settings.")
                st.dataframe(payload_frame(saved["inputs"]["orders"]), use_container_width=True, hide_index=True)
                st.dataframe(payload_frame(saved["artifacts"]["results"]), use_container_width=True, hide_index=True)
                st.download_button("Download saved daily record", json.dumps(saved, ensure_ascii=False, indent=2),
                    f"RAPID_Daily_Record_{selected_date}.json", "application/json")
            else:
                st.info("Save a replan to create the first daily record.")
            st.download_button("Download complete data backup", json.dumps(database.export_records(), ensure_ascii=False, indent=2),
                "RAPID_Data_Backup.json", "application/json")
        except Exception as exc:
            show_storage_error(exc)


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="rapid-brand">
          <div class="rapid-logo">R</div>
          <div><div class="rapid-brand-title">RAPID</div><div class="rapid-brand-sub">RESOURCE ALLOCATION & PRODUCTION INTELLIGENCE FOR DELIVERY</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    nav_labels = ["⌂  Dashboard", "▤  Reports"] + (["⚙  Admin Settings"] if role == "Admin" else [])
    selected = st.radio("Navigation", nav_labels, label_visibility="collapsed")
    page = "Admin Settings" if "Admin Settings" in selected else ("Reports" if "Reports" in selected else "Dashboard")
    st.divider()
    st.markdown("<div class='sidebar-status'><div class='sidebar-status-title'>SYSTEM STATUS</div><div class='status-green'><span class='status-dot'></span>Planning workspace ready</div><div style='font-size:9px;opacity:.65;margin-top:8px'>Draft workspace active</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='signed-label'>SIGNED IN AS</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='signed-user'>{st.session_state.get('username','planner')} · {role}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("↪  SIGN OUT", use_container_width=True):
        logout()
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    st.caption("RAPID · Ocean & Sand")
    st.caption("Local database" if database.is_local else "Shared PostgreSQL database")
    if database.is_local:
        st.caption("Local records survive app restarts on this computer. Configure PostgreSQL for Streamlit Cloud.")
    st.caption("Reload to see another user's saved changes. Unsaved edits remain a draft.")
    if st.button("Reload saved data (discard draft)", use_container_width=True):
        try:
            workspace.reload()
            st.rerun()
        except Exception as exc:
            show_storage_error(exc)
    st.caption("Explainable AI + Rolling Operations Planning")

# -----------------------------------------------------------------------------
# Admin Settings
# -----------------------------------------------------------------------------
if page == "Admin Settings":
    ui_header("Admin Settings", "Configure factory reference values and the process-wise machine master used by RAPID.")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        with st.container(border=True):
            ui_section("Factory Reference Settings", "⚙")
            settings["company_name"] = st.text_input("Company / model label", settings["company_name"], key=f"company_label_{widget_epoch}")
            settings["benchmark_workers"] = st.number_input(
                "Standard Production Workforce",
                min_value=1,
                max_value=1000,
                value=int(settings["benchmark_workers"]),
                step=1,
                help="Reference production workforce. Planner enters actual attendance separately each day.",
                key=f"standard_workers_{widget_epoch}",
            )
            settings["base_capacity"] = st.number_input(
                "Standard Production Capacity (uppers/day)",
                min_value=10.0,
                max_value=10000.0,
                value=float(settings["base_capacity"]),
                step=10.0,
                help="Editable production-capacity benchmark. Replace with verified observed company capacity when available.",
                key=f"standard_capacity_{widget_epoch}",
            )
            settings["max_overtime"] = st.slider(
                "Maximum Overtime Capacity",
                min_value=0.00,
                max_value=0.30,
                value=float(settings["max_overtime"]),
                step=0.01,
                key=f"overtime_limit_{widget_epoch}",
            )
            st.caption("Reference values are prototype assumptions until verified with company records.")
    with c2:
        with st.container(border=True):
            ui_section("Current Reference Snapshot", "◈")
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
            st.dataframe(ref_df, use_container_width=True, hide_index=True, height=178)
            st.info("Workforce and capacity are independent references; changing workforce does not automatically rescale capacity.")

    with st.container(border=True):
        ui_section("Process-wise Machine Master", "▦", "Maintain the machine reference from material cutting through final shoe-upper preparation. Total machines auto-sum from the rows.")
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
            key=f"admin_machine_master_editor_{widget_epoch}",
        )
        st.session_state.machine_master = clean_machine_master(edited_master)
        sync_machine_today()
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Production Processes", len(st.session_state.machine_master))
        mc2.metric("Total Machines", int(st.session_state.machine_master["Total Machines"].sum()))
        mc3.metric("Standard Workforce", int(settings["benchmark_workers"]))

    if st.button("Save Admin Settings", type="primary", use_container_width=True):
        try:
            workspace.save_configuration()
            st.success("Factory settings saved for all users. Planners can reload saved data to use them.")
        except Exception as exc:
            show_storage_error(exc)
    if workspace.configuration_changed():
        st.info("Admin settings have unsaved changes.")

    with st.expander("Explainable Decision Tree & Academic Model Note"):
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_tree(MODEL, feature_names=FEATURES, class_names=[str(c) for c in MODEL.classes_], filled=False, rounded=True, fontsize=7, ax=ax)
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
    ui_header("Reports & Export", "Download the current rolling plan, machine status, material status and management-decision outputs.")
    show_saved_records()
    with st.expander("Production progress & action exports"):
        progress_export = daily_progress(st.session_state.orders, st.session_state.get("production_notes", []), st.session_state.planning_date)
        actions_export = action_frame(st.session_state.get("actions", []))
        st.download_button("Daily Production Progress CSV", progress_export.to_csv(index=False).encode("utf-8"), "RAPID_Production_Progress.csv", "text/csv")
        st.download_button("Corrective Actions CSV", actions_export.to_csv(index=False).encode("utf-8"), "RAPID_Corrective_Actions.csv", "text/csv")
    if results.empty:
        st.info("Add valid active orders on the Dashboard first.")
        st.stop()

    export_results = results.copy()
    export_results["projected_completion_date"] = export_results["completion_day"].apply(lambda x: add_business_days(st.session_state.planning_date, int(x) - 1).date() if pd.notna(x) else pd.NaT)
    export_results = export_results.rename(columns={"projected_delay_days": "projected_delay_days_working"})

    with st.container(border=True):
        ui_section("Export Centre", "⇩")
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.download_button("Order Results CSV", export_results.to_csv(index=False).encode("utf-8"), "RAPID_Order_Results.csv", "text/csv", use_container_width=True)
        with d2:
            st.download_button("Daily Allocation CSV", daily.to_csv(index=False).encode("utf-8"), "RAPID_Daily_Workforce_Allocation.csv", "text/csv", use_container_width=True)
        with d3:
            st.download_button("Machine Status CSV", machine_table.to_csv(index=False).encode("utf-8"), "RAPID_Machine_Status.csv", "text/csv", use_container_width=True)
        with d4:
            st.download_button("Decision Centre CSV", decision_df.to_csv(index=False).encode("utf-8"), "RAPID_Management_Decisions.csv", "text/csv", use_container_width=True)
        e1, e2, e3 = st.columns(3)
        with e1:
            st.download_button("Material Tracker CSV", material_df.to_csv(index=False).encode("utf-8"), "RAPID_Material_Tracker.csv", "text/csv", use_container_width=True)
        with e2:
            history_df = pd.DataFrame(database.history())
            st.download_button("Replan History CSV", history_df.to_csv(index=False).encode("utf-8"), "RAPID_Replan_History.csv", "text/csv", use_container_width=True)
        with e3:
            amap, bottleneck_factor, bottleneck_process, total_m, avail_m, overall_m = machine_availability_map()
            summary_text = f"""RAPID MANAGEMENT SUMMARY\n\nPlanning date: {st.session_state.planning_date}\nWorkers present today: {st.session_state.workers_present}\nStandard workforce: {settings['benchmark_workers']}\nLabour availability factor: {resource_info.get('labour_availability',0):.1%}\nMachines available: {avail_m}/{total_m} ({overall_m:.1%})\nCurrent bottleneck process: {bottleneck_process} ({bottleneck_factor:.1%} available)\nStandard capacity benchmark: {settings['base_capacity']:.0f} uppers/day\nRecommended overtime: {overtime:.1%}\n\nOrder results:\n{export_results.to_string(index=False)}\n\nManagement Decision Centre:\n{decision_df.to_string(index=False)}\n\nAcademic note:\nThe ML model is a proof-of-concept trained on researcher-designed simulated scenarios.\nMachine counts and capacity values are prototype/reference assumptions unless independently verified with company records.\n"""
            st.download_button("Management Summary TXT", summary_text.encode("utf-8"), "RAPID_Management_Summary.txt", "text/plain", use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        with st.container(border=True):
            ui_section("Order Results", "▤")
            st.dataframe(export_results, use_container_width=True, hide_index=True, height=330)
    with r2:
        with st.container(border=True):
            ui_section("Machine Availability by Process", "⚙")
            st.dataframe(machine_table, use_container_width=True, hide_index=True, height=330)
    with st.container(border=True):
        ui_section("Management Decision Centre", "★")
        st.dataframe(decision_df, use_container_width=True, hide_index=True)
    st.stop()

# -----------------------------------------------------------------------------
# Dashboard — single-column management layout
# -----------------------------------------------------------------------------
dashboard_header = st.empty()
st.session_state.orders = normalize_orders(st.session_state.orders)

# Daily controls above the full-width management sections.
with st.container(border=True):
    ui_section("Today's operating plan", "◴", "Update today’s inputs, then save the replan for your team. New dates carry completed production forward and clear daily actuals.")
    c1, c2 = st.columns(2)
    with c1:
        requested_date = st.date_input("Planning Date", value=st.session_state.planning_date, key=f"planning_date_input_{widget_epoch}")
        if requested_date != st.session_state.planning_date:
            try:
                workspace.switch_date(requested_date)
                st.rerun()
            except Exception as exc:
                show_storage_error(exc)
                st.stop()
    with c2:
        max_workers_input = max(500, int(settings["benchmark_workers"]) * 2)
        st.session_state.workers_present = st.number_input(
            "Workers Present Today",
            min_value=0,
            max_value=max_workers_input,
            value=int(st.session_state.workers_present),
            step=1,
            help="Actual production attendance for today. Confirm attendance on every new date.",
            key=f"workers_present_input_{widget_epoch}",
            disabled=st.session_state.get("_historical", False),
        )
    operating_summary = st.empty()
    replan_clicked = st.button("▶  REPLAN TODAY", type="primary", use_container_width=True, disabled=st.session_state.get("_historical", False))
    st.caption("Edits update your draft forecast. REPLAN TODAY saves inputs, daily actuals and the full plan to the database.")

if st.session_state.get("_historical"):
    with dashboard_header.container():
        ui_header("Saved production record", "Read-only history. Select the latest date to continue planning.")
    saved_day = database.daily_record(str(st.session_state.planning_date))
    st.info("This is a saved historical record. It cannot overwrite current production totals.")
    st.dataframe(payload_frame(saved_day["inputs"]["orders"]), use_container_width=True, hide_index=True)
    st.dataframe(payload_frame(saved_day["artifacts"]["results"]), use_container_width=True, hide_index=True)
    render_production_management(st.session_state, read_only=True, key_suffix=str(widget_epoch))
    show_saved_records()
    st.stop()

# Detailed operational inputs are intentionally collapsed so the dashboard stays management-first.
with st.expander("✎  Update today's orders, production, materials & machine status", expanded=False):
    order_tab, machine_tab, progress_tab = st.tabs(["Orders · Production · Materials", "Machine Availability / Breakdown", "Targets · Losses · Actions"])
    with order_tab:
        st.caption("Enter daily production progress, actual workers used, material readiness and each order's current production process. Remaining quantity is calculated automatically.")
        edited_orders = st.data_editor(
            st.session_state.orders,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Order": st.column_config.TextColumn("Order ID", required=True, width="small"),
                "Original Quantity": st.column_config.NumberColumn("Original Qty", min_value=1, step=50, required=True, width="small"),
                "Completed Before Today": st.column_config.NumberColumn("Completed Before", min_value=0, step=10, width="small"),
                "Actual Production Today": st.column_config.NumberColumn("Production Today", min_value=0, step=10, width="small", help="Today's completed output for this order."),
                "Workers Used Today": st.column_config.NumberColumn("Actual Workers Today", min_value=0, step=1, width="small", help="Optional actual worker allocation for Actual vs Planned history."),
                "Due Date": st.column_config.DateColumn("Due Date", required=True, width="small"),
                "Material Status": st.column_config.SelectboxColumn("Material Readiness", options=MATERIAL_STATUSES, required=True, width="medium"),
                "Expected Material Ready Date": st.column_config.DateColumn("Expected Material Ready", width="small"),
                "Current Process": st.column_config.SelectboxColumn("Current Production Process", options=process_options(), required=True, width="medium"),
            },
            key=f"active_orders_editor_{widget_epoch}",
        )
        st.session_state.orders = normalize_orders(edited_orders)
        remaining_view = st.session_state.orders[["Order", "Original Quantity", "Completed Before Today", "Actual Production Today"]].copy()
        remaining_view["Remaining Quantity"] = (remaining_view["Original Quantity"] - remaining_view["Completed Before Today"] - remaining_view["Actual Production Today"]).clip(lower=0)
        st.dataframe(remaining_view, use_container_width=True, hide_index=True, height=185)

    with machine_tab:
        st.caption("Mark available machines by process and record breakdown/maintenance issues. Total process counts come from Admin Settings.")
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
            key=f"machine_today_editor_{widget_epoch}",
        )
        for idx in edited_mt.index:
            total = int(edited_mt.at[idx, "Total Machines"])
            raw_available = pd.to_numeric(edited_mt.at[idx, "Available Today"], errors="coerce")
            available_value = 0 if pd.isna(raw_available) else int(raw_available)
            edited_mt.at[idx, "Available Today"] = max(0, min(total, available_value))
        st.session_state.machine_today = edited_mt
        preview_machine = machine_status_table()
        st.dataframe(preview_machine[["Process", "Total Machines", "Available Today", "Breakdown / Unavailable", "Availability %", "Status", "Breakdown / Issue"]], use_container_width=True, hide_index=True, height=285)
        st.caption("Availability is not the same as true utilization. True utilization requires process run-hours/cycle-time data.")

    with progress_tab:
        if render_production_management(st.session_state, key_suffix=str(widget_epoch)):
            st.rerun()

with dashboard_header.container():
    ui_header("Production Decision Dashboard", "AI/ML-driven planning, resource allocation, process bottleneck visibility & delivery-risk analysis")

# Recalculate after any editor changes.
bundle = calculate_plan()
prepared, pred_df, overtime, results, daily, resource_info, recommendations = bundle
machine_table = machine_status_table()
amap, bottleneck_factor, bottleneck_process, total_machines, available_machines, overall_machine_pct = machine_availability_map()

operating_summary.markdown(
    f"<div class='pill-row'><span class='pill'>Standard workforce: {int(settings['benchmark_workers'])}</span><span class='pill'>Active orders: {len(results)}</span><span class='pill'>Machines available: {available_machines}/{total_machines}</span></div>",
    unsafe_allow_html=True,
)

if replan_clicked:
    try:
        snap = current_snapshot(results, prepared, machine_table)
        workspace.save_plan(snap, {
            "results": frame_payload(results), "daily": frame_payload(daily),
            "predictions": frame_payload(pred_df), "recommendations": frame_payload(recommendations),
            "overtime": float(overtime), "resource_info": resource_info,
        })
        st.success("Plan and daily production records saved for the team.")
    except Exception as exc:
        show_storage_error(exc)

if workspace.operations_changed() or str(st.session_state.planning_date) > st.session_state["_latest_date"]:
    st.info("Draft changes are not saved yet. Use REPLAN TODAY to save them for all users.")
else:
    st.caption(f"Saved plan revision {st.session_state['_ops_revision']} · {st.session_state['_saved_at'][:19].replace('T', ' ')} UTC")

if results.empty:
    st.info("No active quantity remains. You can still save today's actuals and review saved records.")
    show_saved_records()
    st.stop()

missing_ready = [o["order"] for o in prepared if o["material_status"] in BLOCKED_MATERIAL_STATUSES and pd.isna(o.get("expected_material_ready"))]
if missing_ready:
    st.warning(f"Expected Material Ready Date is missing for blocked order(s): {', '.join(missing_ready)}. They remain outside eligible production allocation until a ready date is entered.")

# KPI row
risk_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}
worst_risk = max((str(r).upper() for r in results["risk"]), key=lambda x: risk_rank.get(x, 0))
late_count = int((results["on_time"] == "NO").sum())
if late_count > 0 and worst_risk == "LOW":
    worst_risk = "MEDIUM"
labour_factor = resource_info.get("labour_availability", 0.0)
effective_capacity = resource_info.get("effective_daily_capacity", 0.0)
total_delay = int(results["projected_delay_days"].sum())
on_time = int((results["on_time"] == "YES").sum())
active_orders = len(results)
risk_tone = "green" if worst_risk == "LOW" else ("orange" if worst_risk == "MEDIUM" else "red")
risk_value_class = "risk-low-text" if worst_risk == "LOW" else ("risk-medium-text" if worst_risk == "MEDIUM" else "risk-high-text")

k1, k2, k3 = st.columns(3)
with k1:
    ui_kpi("◆", "OVERALL DELIVERY RISK", worst_risk, f"{late_count} order(s) currently projected late", risk_tone, risk_value_class)
with k2:
    ui_kpi("◴", "AVAILABLE CAPACITY TODAY", f"{effective_capacity:.0f}", "uppers/day after labour, bottleneck & overtime", "blue")
with k3:
    ui_kpi("♟", "WORKERS PRESENT", f"{int(st.session_state.workers_present)}", f"{labour_factor:.0%} of {int(settings['benchmark_workers'])} standard workforce", "violet")
k4, k5, k6 = st.columns(3)
with k4:
    ui_kpi("⚙", "MACHINES AVAILABLE", f"{available_machines}/{total_machines}", f"{bottleneck_process}: {bottleneck_factor:.0%}", "teal")
with k5:
    ui_kpi("✓", "ORDERS ON TIME", f"{on_time}/{active_orders}", "under the current rolling production plan", "green")
with k6:
    ui_kpi("◫", "PROJECTED DELAY", f"{total_delay} days", "total working days across active orders", "red")

st.markdown(
    f"<div class='pill-row'><span class='pill'>Labour availability: {labour_factor:.1%}</span><span class='pill'>Machine availability: {overall_machine_pct:.1%}</span><span class='pill'>Bottleneck factor: {bottleneck_factor:.1%}</span><span class='pill'>Recommended overtime: {overtime:.1%}</span><span class='pill'>Material constraints: {sum(o['material_status'] != 'Ready' for o in prepared)}</span></div>",
    unsafe_allow_html=True,
)

# Production progress complements the forecast; targets are explicit daily commitments.
with st.container(border=True):
    ui_section("Daily progress & follow-up", "↗", "Record targets, reasons and action owners in the Targets · Losses · Actions input tab.")
    progress = daily_progress(st.session_state.orders, st.session_state.get("production_notes", []), st.session_state.planning_date)
    followups = action_summary(st.session_state.get("actions", []), st.session_state.planning_date)
    st.markdown(f"**{followups['open']} open actions** · {followups['overdue']} overdue · {followups['follow_up_due']} follow-ups due")
    st.dataframe(progress[["Order", "Daily Target", "Good Output Today", "Target Remaining", "Target Reached (%)", "Progress", "Recorded Reason"]], use_container_width=True, hide_index=True)
    st.caption("Daily targets are entered by the supervisor. A remaining target is not a delivery-delay prediction; recorded reasons are observations.")

# Risk + allocation
with st.container(border=True):
    ui_section("AI/ML Delivery-Risk Assessment", "◈")
    risk_display = pred_df.copy()
    st.dataframe(risk_display, use_container_width=True, hide_index=True, height=min(400, 38 + 35 * len(risk_display)))
    st.caption("LOW = manageable · MEDIUM = warning · HIGH = serious delivery risk. Prototype Model Confidence is not certainty of actual delivery outcome.")
with st.container(border=True):
    ui_section("Resource Allocation Summary", "⌘")
    alloc = results[["order", "day1_workers", "current_process", "completion_day", "projected_delay_days", "on_time"]].copy()
    alloc["Projected Completion"] = alloc["completion_day"].apply(lambda x: add_business_days(st.session_state.planning_date, int(x) - 1).strftime("%d %b %Y") if pd.notna(x) else "—")
    alloc = alloc[["order", "day1_workers", "current_process", "Projected Completion", "projected_delay_days", "on_time"]]
    alloc.columns = ["Order", "Recommended Workers Today", "Primary Process Today", "Projected Completion", "Projected Delay (Days)", "On Time?"]
    st.dataframe(alloc, use_container_width=True, hide_index=True, height=min(400, 38 + 35 * len(alloc)))
    with st.expander("View process-wise worker allocation"):
        st.dataframe(build_process_allocation(results), use_container_width=True, hide_index=True)
        st.caption("Workers are assigned to the planner-selected current production process; the prototype does not falsely allocate the same worker across multiple stages simultaneously.")

with st.expander("Explain an order · evidence and recommended action"):
    explain_order = st.selectbox("Order to explain", [o["order"] for o in prepared])
    evidence = next(o for o in prepared if o["order"] == explain_order)
    outcome = results.loc[results["order"] == explain_order].iloc[0]
    advice = recommendations.loc[recommendations["order"] == explain_order].iloc[0]
    st.write(f"Remaining quantity: {evidence['quantity']:,}. Recommended workers today: {int(outcome['day1_workers'])}. Current process: {evidence['current_process']}.")
    st.write(f"Machine availability: {evidence['machine_availability']:.0%}. Material status: {evidence['material_status']}. Delivery assessment: {advice['status']}.")
    st.markdown("**Recorded constraints and operational indicators**")
    st.write(advice["reason"])
    st.markdown("**Recommended action**")
    st.write(advice["recommended_action"])
    st.caption("Allocation uses material readiness, usable process capacity, due-date priority and ML risk. Shared workforce is limited to the configured attendance basis. Test changes in What-if Impact Analysis before committing them.")
    feature_values = pd.DataFrame([{
        "quantity": float(evidence["quantity"]), "days_remaining": evidence["days_remaining"],
        "machine_availability": evidence["machine_availability"], "labour_availability": labour_factor,
        "material_delay_days": evidence["ml_material_delay_days"], "base_capacity": float(settings["base_capacity"]),
        "required_daily_output": evidence["required_daily_output"], "capacity_ratio": evidence["capacity_ratio"],
    }], columns=FEATURES)
    decision_nodes = MODEL.decision_path(feature_values).indices
    tree_evidence = []
    for node in decision_nodes:
        feature_index = int(MODEL.tree_.feature[node])
        if feature_index >= 0:
            observed = float(feature_values.iloc[0, feature_index])
            threshold = float(MODEL.tree_.threshold[node])
            tree_evidence.append({"Model input": FEATURES[feature_index], "Observed value": observed,
                "Condition followed": "≤" if observed <= threshold else ">", "Tree threshold": threshold})
    st.markdown("**Actual ML decision path**")
    st.dataframe(pd.DataFrame(tree_evidence), use_container_width=True, hide_index=True)
    st.caption("These are the classifier's actual tests for this order. The model uses simulated training scenarios; the decision path explains its prediction, not a proven factory root cause. This is plan guidance, not a generative chatbot.")

# Machine and material visibility
with st.container(border=True):
    ui_section("Machine Availability & Bottleneck View", "⚙", "Process-wise machine visibility makes breakdown location and production bottlenecks explicit.")
    machine_display = machine_table[["Process", "Total Machines", "Available Today", "Breakdown / Unavailable", "Availability %", "Status"]].copy()
    machine_display["Availability %"] = machine_display["Availability %"].round(1)
    st.dataframe(machine_display, use_container_width=True, hide_index=True, height=min(460, 38 + 35 * len(machine_display)))
    st.caption(f"Current bottleneck: {bottleneck_process} ({bottleneck_factor:.1%} available).")
with st.container(border=True):
    ui_section("Material Readiness & Constraint Tracker", "▧", "Blocked/held material removes an order from eligible production allocation until the ready condition is restored.")
    material_df = build_material_tracker(prepared)
    st.dataframe(material_df, use_container_width=True, hide_index=True, height=min(400, 38 + 35 * len(material_df)))

# Daily Change Monitor, rendered as compact cards.
change_df = change_monitor_df(st.session_state.plan_history)
with st.container(border=True):
    ui_section("Daily Change Monitor", "↺", "Previous-plan vs today's condition changes after rolling replans.")
    if change_df.empty:
        st.info("Run REPLAN TODAY on at least two planning updates to activate previous-vs-today comparison.")
    else:
        st.dataframe(change_df, use_container_width=True, hide_index=True)

# Workforce allocation + Management Decision Centre
with st.container(border=True):
    ui_section("Daily Workforce Allocation by Order", "⌁")
    fig = plot_daily_workforce(daily, st.session_state.plan_history)
    if fig is not None:
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    st.caption("Production Date × Workers Allocated. Actual worker use is retained in replan history when entered; future allocation is the revised RAPID recommendation.")
with st.container(border=True):
    ui_section("Management Decision Centre", "★")
    decision_df = management_decision_rows(results, prepared, labour_factor, machine_table)
    act_now = int((decision_df["Priority"] == "🔴 ACT NOW").sum())
    action_today = int((decision_df["Priority"] == "🟠 ACTION TODAY").sum())
    if act_now:
        st.markdown(f"<div class='action-banner action-red'>⚠ ACTION REQUIRED · {act_now} critical management action(s) identified.</div>", unsafe_allow_html=True)
    elif action_today:
        st.markdown(f"<div class='action-banner action-amber'>Attention needed today · {action_today} corrective action(s) identified.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='action-banner action-green'>✓ No immediate corrective action is required under the current plan.</div>", unsafe_allow_html=True)
    st.dataframe(decision_df[["Priority", "Issue", "Affected", "Recommended Action", "When"]].head(6), use_container_width=True, hide_index=True, height=38 + 35 * min(6, len(decision_df)))
    with st.expander("Why · Expected Impact · If No Action"):
        st.dataframe(decision_df, use_container_width=True, hide_index=True)

# What-if impact analysis
with st.container(border=True):
    ui_section("What-if Impact Analysis", "⌁", "Hypothetical simulations only. Scenarios compare possible disruptions against today's Current Operating Plan and do not modify the active plan.")
    scenario_df, scenario_configs = evaluate_scenarios(bundle, machine_table)
    scenario_cards = scenario_df.head(6).to_dict("records")
    for sc in scenario_cards:
        impact_colour = "#20745B" if "STABLE" in sc["Impact"] else ("#946013" if "WATCH" in sc["Impact"] or "WARNING" in sc["Impact"] else "#AD3E40")
        st.markdown(
            f"<div class='scenario-card'><div class='scenario-name'>{sc['Scenario']}</div><div class='scenario-impact' style='color:{impact_colour}'>{sc['Impact']}</div><div class='scenario-big'>{sc['Orders On Time']}</div><div class='scenario-small'>orders on time · {sc['Projected Delay (Days)']} projected delay day(s)<br>{sc['Comparison']}</div></div>",
            unsafe_allow_html=True,
        )

    with st.expander("View assumptions, affected orders, reasons & suggested responses"):
        st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    with st.expander("Test Recovery Action"):
        non_baseline = [s for s in scenario_df["Scenario"].tolist() if s != "Current Operating Plan"]
        rc1, rc2 = st.columns(2)
        with rc1:
            selected_scenario = st.selectbox("Scenario to recover", non_baseline)
        with rc2:
            recovery_action = st.selectbox("Recovery action to test", ["Use Maximum Overtime", "Recover 50% of Machine Loss", "Recover 10% of Standard Workforce", "Expedite Material by 2 Working Days"])
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

            after_bundle = calculate_plan(workers_override=workers, machine_map_override=m_map, extra_delay_map=extra_delay, status_override_map=status_override, overtime_override=overtime_override)
            before_res = before_bundle[3]
            after_res = after_bundle[3]
            before_on = int((before_res["on_time"] == "YES").sum())
            after_on = int((after_res["on_time"] == "YES").sum())
            before_delay = int(before_res["projected_delay_days"].sum())
            after_delay = int(after_res["projected_delay_days"].sum())
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Before · Orders On Time", f"{before_on}/{len(before_res)}")
            r2.metric("After · Orders On Time", f"{after_on}/{len(after_res)}", delta=after_on - before_on)
            r3.metric("Before · Projected Delay", before_delay)
            r4.metric("After · Projected Delay", after_delay, delta=after_delay - before_delay, delta_color="inverse")
            st.caption("Recovery-test results are recalculated by the planning engine and remain decision-support estimates, not guaranteed outcomes.")

with st.expander("Management detail · order reasons · process allocation · full rolling schedule"):
    st.markdown("**Order-level recommendations**")
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    st.markdown("**Process-wise worker allocation today**")
    st.dataframe(build_process_allocation(results), use_container_width=True, hide_index=True)
    st.markdown("**Full planned daily allocation**")
    st.dataframe(daily, use_container_width=True, hide_index=True, height=360)

st.markdown(
    "<div class='academic-note'><b>RAPID</b> — Resource Allocation & Production Intelligence for Delivery · MBA decision-support prototype. Academic disclosure: ML training scenarios are researcher-designed/simulated; default workforce, machine counts and capacity values are prototype/reference assumptions unless verified with company records. Saved records are stored in the configured database; edits remain drafts until saved. Local SQLite is for local use; cloud persistence requires PostgreSQL.</div>",
    unsafe_allow_html=True,
)
