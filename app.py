
from pathlib import Path
from datetime import date
import io
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
FEATURES = ['quantity', 'days_remaining', 'machine_availability', 'labour_availability', 'material_delay_days', 'base_capacity', 'required_daily_output', 'capacity_ratio']

st.set_page_config(page_title="SURE-Plan", page_icon="🏭", layout="wide")

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
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def login_page():
    st.title("🏭 SURE-Plan")
    st.subheader("Dynamic Resource Reallocation & Delivery-Risk Decision Support System")
    st.caption("Make-to-order shoe upper production | MBA project prototype")

    c1, c2, c3 = st.columns([1, 1.1, 1])
    with c2:
        st.markdown("### Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Sign in", type="primary", use_container_width=True):
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
        st.info("Admin manages benchmark resources. User/Planner enters orders and runs production plans.")

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
        {"Order":"A","Quantity":2000,"Due Date":pd.Timestamp("2026-09-30"),"Material Delay (days)":0},
        {"Order":"B","Quantity":1500,"Due Date":pd.Timestamp("2026-09-15"),"Material Delay (days)":0},
        {"Order":"C","Quantity":1100,"Due Date":pd.Timestamp("2026-09-10"),"Material Delay (days)":0},
    ])

# Header
h1, h2 = st.columns([5,1])
with h1:
    st.title("🏭 SURE-Plan")
    st.caption("From order intake to delivery-risk prediction, resource allocation and corrective-action recommendation")
with h2:
    st.write(f"**{role}**")
    if st.button("Logout", use_container_width=True):
        logout()

st.sidebar.header("Current Operating Condition")
planning_date = st.sidebar.date_input("Planning date", value=date(2026, 9, 1))
machine_availability = st.sidebar.slider("Machine availability", 0.50, 1.00, 1.00, 0.01)
labour_availability = st.sidebar.slider("Labour availability", 0.50, 1.00, 1.00, 0.01)

settings = st.session_state.settings

if role == "Admin":
    st.sidebar.divider()
    st.sidebar.header("🔐 Admin Resource Settings")
    settings["company_name"] = st.sidebar.text_input("Company/model label", settings["company_name"])
    settings["base_capacity"] = st.sidebar.number_input(
        "Benchmark production capacity (uppers/day)",
        min_value=50.0, max_value=3000.0,
        value=float(settings["base_capacity"]), step=10.0
    )
    settings["benchmark_workers"] = st.sidebar.number_input(
        "Benchmark production workers",
        min_value=1, max_value=500,
        value=int(settings["benchmark_workers"]), step=1
    )
    settings["max_overtime"] = st.sidebar.slider(
        "Maximum overtime capacity",
        0.00, 0.30, float(settings["max_overtime"]), 0.01
    )
    st.sidebar.warning("Benchmark values are model/secondary assumptions unless verified with company records.")
else:
    st.sidebar.info("Core resource assumptions are controlled by Admin.")

# Tabs = full A-Z application
tab_names = [
    "1️⃣ Project Context",
    "2️⃣ Order Intake",
    "3️⃣ AI/ML Risk",
    "4️⃣ Resource Allocation",
    "5️⃣ Daily Schedule",
    "6️⃣ Disruption Lab",
    "7️⃣ Solution Centre",
    "8️⃣ Export",
]
if role == "Admin":
    tab_names.append("9️⃣ Admin & Model")

tabs = st.tabs(tab_names)
context_tab, intake_tab, risk_tab, allocation_tab, schedule_tab, scenario_tab, solution_tab, export_tab = tabs[:8]
admin_tab = tabs[8] if role == "Admin" else None

with context_tab:
    st.header("Project Context")
    st.write(
        "SURE-Plan is a decision-support prototype for concurrent make-to-order shoe-upper production. "
        "Its purpose is to identify which active orders are at delivery risk when production is disrupted, "
        "recommend how limited production capacity should be shared, and show the effect of that decision on all active orders."
    )
    st.markdown("""
**A–Z workflow**

**Orders received → Due-date requirement → Current machine/labour/material condition → Explainable ML risk → Capacity requirement → Dynamic worker allocation → Projected completion → Disruption testing → Corrective action recommendation**
""")
    st.warning(
        "Academic disclosure: the current ML layer uses researcher-designed simulated scenarios and model/secondary benchmark assumptions. "
        "It is a proof-of-concept and should not be presented as validated historical Keerthi Shoe Fabrics data."
    )

with intake_tab:
    st.header("Concurrent Order Intake")
    st.caption("Enter every active make-to-order shoe-upper order being processed simultaneously.")
    st.session_state.orders = st.data_editor(
        st.session_state.orders,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Order": st.column_config.TextColumn(required=True),
            "Quantity": st.column_config.NumberColumn(min_value=1, step=50, required=True),
            "Due Date": st.column_config.DateColumn(required=True),
            "Material Delay (days)": st.column_config.NumberColumn(min_value=0, step=1),
        },
    )
    st.info("Material Delay means the order cannot start/continue normally until the required material becomes available.")

# Prepare model inputs
prepared = []
prediction_rows = []

for _, row in st.session_state.orders.iterrows():
    if pd.isna(row.get("Order")) or pd.isna(row.get("Quantity")) or pd.isna(row.get("Due Date")):
        continue

    order = str(row["Order"])
    qty = float(row["Quantity"])
    days = business_days_between(planning_date, row["Due Date"])
    mat_delay = int(row.get("Material Delay (days)", 0) or 0)
    effective_days = max(1, days - mat_delay)
    req_daily = qty / max(1, days)
    req_after_delay = qty / effective_days
    effective_cap = settings["base_capacity"] * machine_availability * labour_availability
    ratio = req_after_delay / max(effective_cap, 1e-9)

    x = pd.DataFrame([{
        "quantity": qty,
        "days_remaining": days,
        "machine_availability": machine_availability,
        "labour_availability": labour_availability,
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
        "ML Confidence %": round(conf, 1),
    })

overtime = recommend_overtime(
    prepared,
    settings["base_capacity"],
    machine_availability,
    labour_availability,
    settings["max_overtime"],
) if prepared else 0.0

results, daily, resource_info = simulate_orders(
    prepared,
    settings["base_capacity"],
    settings["benchmark_workers"],
    machine_availability,
    labour_availability,
    overtime,
) if prepared else (pd.DataFrame(), pd.DataFrame(), {})

recommendations = build_recommendations(
    results, prepared, machine_availability, labour_availability, overtime
) if not results.empty else pd.DataFrame()

with risk_tab:
    st.header("Explainable AI/ML Delivery-Risk Assessment")
    if prediction_rows:
        pred_df = pd.DataFrame(prediction_rows)
        st.dataframe(pred_df, use_container_width=True)
        st.caption(
            "LOW / MEDIUM / HIGH is predicted using order load, due-date pressure, machine availability, "
            "labour availability, material delay and capacity ratio."
        )
    else:
        st.info("Add at least one valid order.")

with allocation_tab:
    st.header("Dynamic Resource Allocation")
    if not results.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Workers available", resource_info["available_workers"])
        c2.metric("Effective capacity/day", f'{resource_info["effective_daily_capacity"]:.0f}')
        c3.metric("Recommended overtime", f"{overtime*100:.1f}%")
        c4.metric("Orders on time", f'{int((results["on_time"]=="YES").sum())}/{len(results)}')

        allocation = results[[
            "order","quantity","risk","required_daily_output",
            "day1_workers","completion_day","tardiness_days","on_time"
        ]].copy()
        allocation.columns = [
            "Order","Quantity","ML Risk","Required Daily Output",
            "Recommended Day-1 Workers","Projected Completion Working Day",
            "Projected Tardiness","On Time"
        ]
        st.dataframe(allocation, use_container_width=True)

        st.info(
            "The allocation engine gives priority to orders with tighter due dates, larger capacity shortfalls and higher ML risk, "
            "while still checking the impact on the other concurrent orders."
        )

with schedule_tab:
    st.header("Dynamic Daily Resource Schedule")
    if not daily.empty:
        pivot = daily.pivot(index="day", columns="order", values="workers_allocated").fillna(0)
        st.line_chart(pivot)
        st.dataframe(daily, use_container_width=True, height=380)
        st.caption("The worker allocation changes as orders finish or material becomes available.")

with scenario_tab:
    st.header("Disruption / What-if Lab")
    scenario_defs = [
        ("Normal", 1.00, 1.00, 0),
        ("Machine breakdown", 0.80, 1.00, 0),
        ("Labour shortage", 1.00, 0.80, 0),
        ("Material delay on Order C", 1.00, 1.00, 3),
        ("Combined disruption", 0.85, 0.90, 2),
        ("Severe combined disruption", 0.75, 0.80, 4),
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
            ratio = (
                x_o["quantity"] / effective_days
            ) / max(settings["base_capacity"] * m * l, 1e-9)

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

        sc_ot = recommend_overtime(
            temp_orders, settings["base_capacity"], m, l, settings["max_overtime"]
        ) if temp_orders else 0

        sc_res, _, _ = simulate_orders(
            temp_orders, settings["base_capacity"], settings["benchmark_workers"],
            m, l, sc_ot
        ) if temp_orders else (pd.DataFrame(), pd.DataFrame(), {})

        scenario_output.append({
            "Scenario": name,
            "Machine Availability": f"{m:.0%}",
            "Labour Availability": f"{l:.0%}",
            "Recommended Overtime": f"{sc_ot:.1%}",
            "Orders On Time": int((sc_res["on_time"]=="YES").sum()) if not sc_res.empty else 0,
            "Total Orders": len(sc_res),
            "Total Tardiness Days": int(sc_res["tardiness_days"].sum()) if not sc_res.empty else 0,
        })

    sc_df = pd.DataFrame(scenario_output)
    st.dataframe(sc_df, use_container_width=True)
    if not sc_df.empty:
        chart_df = sc_df.set_index("Scenario")[["Total Tardiness Days"]]
        st.bar_chart(chart_df)

with solution_tab:
    st.header("Final Solution Centre")
    if not recommendations.empty:
        for _, row in recommendations.iterrows():
            if row["status"] == "LATE":
                st.error(f'Order {row["order"]} — {row["risk"]} Risk — Projected Late')
            elif row["risk"] == "MEDIUM":
                st.warning(f'Order {row["order"]} — Medium Risk — On-time currently, but needs monitoring')
            else:
                st.success(f'Order {row["order"]} — Low Risk — Current plan acceptable')

            st.write(f'**Why:** {row["reason"]}')
            st.write(f'**Recommended action:** {row["recommended_action"]}')
            st.divider()

        st.subheader("Overall Management Decision")
        late_count = int((results["on_time"]=="NO").sum())
        if late_count == 0:
            st.success(
                "The current resource plan is feasible under the selected assumptions. "
                "Continue the recommended allocation and use the disruption triggers for daily monitoring."
            )
        else:
            st.error(
                f"{late_count} order(s) remain at risk of late completion even after dynamic allocation. "
                "Management should combine resource reallocation with additional capacity actions such as overtime, "
                "maintenance recovery, material expediting, subcontracting or customer due-date renegotiation."
            )

with export_tab:
    st.header("Export Results")
    if not results.empty:
        export_results = results.copy()
        export_results["projected_completion_date"] = export_results["completion_day"].apply(
            lambda x: add_business_days(planning_date, int(x)).date()
        )

        st.download_button(
            "Download Order Results CSV",
            export_results.to_csv(index=False).encode("utf-8"),
            file_name="SURE_Plan_Order_Results.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Daily Allocation CSV",
            daily.to_csv(index=False).encode("utf-8"),
            file_name="SURE_Plan_Daily_Allocation.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download Recommendations CSV",
            recommendations.to_csv(index=False).encode("utf-8"),
            file_name="SURE_Plan_Recommendations.csv",
            mime="text/csv",
        )

        summary_text = f"""SURE-PLAN MANAGEMENT SUMMARY

Planning date: {planning_date}
Machine availability: {machine_availability:.0%}
Labour availability: {labour_availability:.0%}
Benchmark capacity: {settings["base_capacity"]:.0f} uppers/day
Benchmark workers: {settings["benchmark_workers"]}
Recommended overtime: {overtime:.1%}

Orders:
{export_results.to_string(index=False)}

Recommendations:
{recommendations.to_string(index=False)}

Academic note:
The current ML model is a proof-of-concept trained on researcher-designed simulated scenarios.
Benchmark resource values are model/secondary-data assumptions unless independently verified.
"""
        st.download_button(
            "Download Management Summary TXT",
            summary_text.encode("utf-8"),
            file_name="SURE_Plan_Management_Summary.txt",
            mime="text/plain",
        )

if role == "Admin" and admin_tab is not None:
    with admin_tab:
        st.header("Admin & Model Control")
        st.subheader("Current benchmark assumptions")
        st.json(settings)

        st.subheader("Explainable Decision Tree")
        fig, ax = plt.subplots(figsize=(15,8))
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

        st.write("**Prototype hold-out classification accuracy on simulated data:** 99.9%")
        st.warning(
            "This percentage is validation on researcher-designed simulated data, not historical company validation."
        )

st.divider()
st.caption("SURE-Plan MBA prototype | Explainable ML + operations planning + disruption-aware resource reallocation")
