"""Planner-facing views over existing results; no scheduling or ML changes."""
import math
import pandas as pd
from planner_core import add_business_days, validate_order_ids


def allocation_view(orders, daily):
    """Use workers and output from the SAME first scheduled date."""
    columns = ["Order", "Process", "Workers", "Expected output", "Order balance", "Plan date"]
    if daily.empty:
        return pd.DataFrame(columns=columns)
    first = daily.loc[daily["production_date"] == daily["production_date"].min()]
    validate_order_ids(first["order"])
    balances = {
        str(r["Order"]): max(0, int(r["Original Quantity"]) - int(r["Completed Before Today"]) - int(r["Actual Production Today"]))
        for _, r in orders.iterrows()
    }
    return pd.DataFrame([{
        "Order": str(r["order"]), "Process": r["current_process"],
        "Workers": int(r["workers_allocated"]),
        "Expected output": math.floor(max(0.0, float(r["produced"]))),
        "Order balance": balances.get(str(r["order"]), 0),
        "Plan date": pd.Timestamp(r["production_date"]).date(),
    } for _, r in first.iterrows()], columns=columns)


def overview_chart_data(progress, allocation):
    """Keep daily targets/actuals separate from recommended process staffing."""
    output = progress.loc[progress["Daily Target"].notna(),
        ["Order", "Daily Target", "Good Output Today"]].copy()
    for column in ("Daily Target", "Good Output Today"):
        output[column] = pd.to_numeric(output[column], errors="raise").astype(int)
    output = output.loc[(output["Daily Target"] > 0) | (output["Good Output Today"] > 0)]
    staffing = allocation.groupby("Process", as_index=False, sort=False)["Workers"].sum()
    staffing = staffing.loc[staffing["Workers"] > 0].sort_values("Workers", ascending=False)
    return output, staffing


def confirm_staffing(orders, allocation, attendance, planning_date):
    """Record planner-confirmed assignments, without changing output or targets."""
    validate_order_ids(orders["Order"])
    validate_order_ids(allocation["Order"])
    if allocation.empty:
        raise ValueError("There is no allocation to confirm.")
    if any(pd.Timestamp(d).date() != pd.Timestamp(planning_date).date() for d in allocation["Plan date"]):
        raise ValueError("This allocation is for the next working date. Confirm staffing on that date.")
    values = pd.to_numeric(allocation["Workers"], errors="coerce")
    if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
        raise ValueError("Worker assignments must be non-negative whole numbers.")
    if values.sum() > int(attendance):
        raise ValueError("Worker assignments exceed today's attendance.")
    if not set(allocation["Order"]).issubset(set(orders["Order"])):
        raise ValueError("The order list has changed. Recalculate the worker plan.")
    result = orders.copy(deep=True)
    assigned = dict(zip(allocation["Order"], values.astype(int)))
    result["Workers Used Today"] = result["Order"].map(assigned).fillna(0).astype(int)
    return result


def delivery_view(orders, results, planning_date):
    """Keep unknown completion and lower-bound delay explicit in the UI."""
    rows = []
    indexed = {r["order"]: r for _, r in results.iterrows()} if not results.empty else {}
    for _, order in orders.iterrows():
        balance = max(0, int(order["Original Quantity"]) - int(order["Completed Before Today"]) - int(order["Actual Production Today"]))
        result = indexed.get(order["Order"])
        complete = result is not None and pd.notna(result["completion_day"])
        forecast = add_business_days(planning_date, int(result["completion_day"]) - 1).strftime("%d %b %Y") if complete else "Not forecast"
        if balance == 0:
            status, forecast, delay = "Quantity complete", "Completion date not recorded", "—"
        elif result is None:
            status, delay = "Check order inputs", "Unknown"
        else:
            status = {"YES": "On schedule", "NO": "Late forecast", "UNKNOWN": "Unconfirmed"}.get(result["on_time"], "Unconfirmed")
            if not complete:
                status = "No completion within horizon"
            days = int(result["projected_delay_days"])
            unit = "working day" if days == 1 else "working days"
            delay = f"At least {days} {unit}" if result.get("projected_delay_is_lower_bound", False) else f"{days} {unit}"
        rows.append({"Order": order["Order"], "Quantity left": balance,
                     "Due date": pd.Timestamp(order["Due Date"]).strftime("%d %b %Y") if pd.notna(order["Due Date"]) else "Missing",
                     "Forecast completion": forecast, "Delivery status": status, "Delay": delay})
    return pd.DataFrame(rows, columns=["Order", "Quantity left", "Due date", "Forecast completion", "Delivery status", "Delay"])
