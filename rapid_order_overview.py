"""A quantity-based order overview over the existing planning results."""
from html import escape

import pandas as pd
import streamlit as st


def completion_parts(order):
    total = int(order["Original Quantity"])
    before = int(order["Completed Before Today"])
    today = int(order["Actual Production Today"])
    if total <= 0 or min(before, today) < 0 or before + today > total:
        raise ValueError("Check this order's quantities in Setup inputs: completed units must be between zero and the total order quantity.")
    return total, before, today, total - before - today


def render_order_overview(orders, progress, allocation, delivery, planning_date):
    st.subheader("Order overview")
    st.caption("Choose an order to see its quantity completion, today's progress and latest delivery forecast.")
    if orders.empty:
        st.info("Add an order in Setup inputs to see its overview.")
        return
    ids = orders["Order"].tolist()
    if st.session_state.get("overview_order") not in ids:
        st.session_state["overview_order"] = ids[0]
    selected = st.selectbox("Order to review", ids, key="overview_order")
    order = orders.loc[orders["Order"] == selected].iloc[0]
    with st.container(border=True):
        st.markdown(f"### Order {selected}")
        st.caption(f"Current process: {order['Current Process']} · Planning date: {planning_date:%d %b %Y}")
        try:
            total, before, today, remaining = completion_parts(order)
        except ValueError as exc:
            st.error(str(exc))
            return
        completed = before + today
        percent = completed / total * 100
        first, second = before / total * 100, completed / total * 100
        parts = [("Completed before today", before, "#103c4a"),
                 ("Produced today", today, "#087e8b"),
                 ("Remaining quantity", remaining, "#d8bb8b")]
        legend = "".join(f'<div class="order-legend-row"><span class="order-dot" style="background:{color}"></span><div><span>{label}</span><strong>{value:,} <small>units · {value / total:.1%}</small></strong></div></div>' for label, value, color in parts)
        description = escape(f"Order {selected}: {before} completed before today, {today} produced today, {remaining} remaining out of {total} units.", quote=True)
        st.markdown(f'<div class="order-visual"><div class="order-ring" role="img" aria-label="{description}" style="background:conic-gradient(#103c4a 0% {first}%,#087e8b {first}% {second}%,#d8bb8b {second}% 100%)"><div class="order-ring-centre"><strong>{percent:.1f}%</strong><span>COMPLETE</span><small>{completed:,} of {total:,} units</small></div></div><div class="order-legend"><div class="order-legend-title">QUANTITY BREAKDOWN</div>{legend}</div></div>', unsafe_allow_html=True)
        st.caption("Completion uses recorded good units. The current process is shown above; the ring does not imply that every production stage has been completed.")
        day = progress.loc[progress["Order"] == selected]
        target = day.iloc[0]["Daily Target"] if not day.empty else None
        target_known = target is not None and pd.notna(target)
        a, b, c = st.columns(3)
        a.metric("Today's target", f"{int(target):,} units" if target_known else "Not set")
        b.metric("Produced today", f"{today:,} units")
        c.metric("Target left today", f"{max(0, int(target)-today):,} units" if target_known else "Not set")
        staff = allocation.loc[allocation["Order"] == selected]
        workers = int(staff.iloc[0]["Workers"]) if not staff.empty else 0
        staffing_date = staff.iloc[0]["Plan date"] if not staff.empty else planning_date
        forecast = delivery.loc[delivery["Order"] == selected].iloc[0]
        a, b, c = st.columns(3)
        a.markdown("**Recommended workers**")
        a.write(str(workers))
        b.markdown("**Customer due date**")
        b.write(forecast["Due date"])
        c.markdown("**Forecast completion**")
        c.write(forecast["Forecast completion"])
        if staffing_date != planning_date:
            st.caption(f"The worker recommendation is for the next working date: {staffing_date:%d %b %Y}.")
        status = str(forecast["Delivery status"])
        status_text = f"Delivery status: {status} · Delay: {forecast['Delay']}"
        if status == "Late forecast" or status == "No completion within horizon":
            st.warning(status_text)
        elif status in ("On schedule", "Quantity complete"):
            st.success(status_text)
        else:
            st.info(status_text)
        if target_known and int(target) > 0:
            daily_text = (f"Today's target is reached ({today:,} of {int(target):,} units)." if today >= int(target)
                          else f"Today, {today:,} of {int(target):,} target units are recorded, leaving {int(target)-today:,} units to reach the target.")
        else:
            daily_text = "No output target is scheduled for this date." if target_known else "Today's target has not been set."
        st.markdown("**What this means**")
        st.write(f"Order {selected} is {percent:.1f}% complete, with {remaining:,} units remaining. {daily_text}")
        st.caption("Record actual output in Production progress. Staffing and delivery forecasts update with the current plan; saved daily targets remain fixed unless adjusted. A target shortfall during the day does not by itself mean delivery is late.")
