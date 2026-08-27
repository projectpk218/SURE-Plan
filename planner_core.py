
import math
import pandas as pd

RISK_MULTIPLIER = {"LOW": 1.00, "MEDIUM": 1.18, "HIGH": 1.40}

def business_days_between(start_date, due_date):
    start = pd.Timestamp(start_date).normalize()
    due = pd.Timestamp(due_date).normalize()
    if due < start:
        return 1
    return max(1, len(pd.bdate_range(start, due)))

def add_business_days(start_date, working_days):
    d = pd.Timestamp(start_date)
    if working_days <= 0:
        return d
    return pd.bdate_range(d, periods=working_days + 1)[-1]

def recommend_overtime(orders, base_capacity, machine_availability, labour_availability, max_overtime):
    required = sum(
        float(o["quantity"]) / max(1, int(o["days_remaining"]))
        for o in orders
    )
    regular_capacity = base_capacity * machine_availability * labour_availability
    needed = max(0.0, required / max(regular_capacity, 1e-9) - 1.0)
    return min(max_overtime, needed)

def allocate_workers_for_day(states, available_workers, capacity_per_worker, day):
    ready = [
        s for s in states
        if s["remaining"] > 1e-9 and day >= s["material_delay_days"]
    ]
    allocation = {s["order"]: 0 for s in states}
    if not ready:
        return allocation

    # Urgency = capacity requirement adjusted by ML risk and due-date pressure.
    scored = []
    for s in ready:
        days_left = max(1, s["due_day"] - day)
        required_workers = (
            s["remaining"] / days_left / max(capacity_per_worker, 1e-9)
        )
        priority = (
            required_workers
            * RISK_MULTIPLIER.get(s["risk"], 1.0)
            * (1 + 1 / days_left)
        )
        scored.append((s, required_workers, priority))

    # First keep each ready order alive where capacity allows.
    scored.sort(key=lambda x: (x[0]["due_day"] - day, -x[2]))
    remaining_workers = available_workers
    for s, _, _ in scored:
        if remaining_workers <= 0:
            break
        allocation[s["order"]] = 1
        remaining_workers -= 1

    # Allocate remaining workers to the largest weighted shortfall.
    while remaining_workers > 0 and scored:
        best = None
        best_gap = -1e18
        for s, required_workers, priority in scored:
            allocated = allocation[s["order"]]
            gap = priority - allocated
            if gap > best_gap:
                best_gap = gap
                best = s
        allocation[best["order"]] += 1
        remaining_workers -= 1

    return allocation

def simulate_orders(
    orders,
    base_capacity,
    benchmark_workers,
    machine_availability,
    labour_availability,
    overtime_fraction,
    horizon=120,
):
    available_workers = max(1, math.floor(benchmark_workers * labour_availability))
    capacity_per_worker = (
        base_capacity / benchmark_workers
        * machine_availability
        * (1 + overtime_fraction)
    )

    states = []
    for o in orders:
        states.append({
            **o,
            "remaining": float(o["quantity"]),
            "completion_day": None,
        })

    daily = []

    for day in range(horizon):
        if all(s["remaining"] <= 1e-9 for s in states):
            break

        allocation = allocate_workers_for_day(
            states, available_workers, capacity_per_worker, day
        )

        for s in states:
            ready = day >= s["material_delay_days"]
            workers = allocation.get(s["order"], 0)
            produced = 0.0

            if ready and workers > 0 and s["remaining"] > 0:
                produced = min(
                    s["remaining"],
                    workers * capacity_per_worker,
                )
                s["remaining"] -= produced

                if s["remaining"] <= 1e-9 and s["completion_day"] is None:
                    s["completion_day"] = day + 1

            daily.append({
                "day": day + 1,
                "order": s["order"],
                "risk": s["risk"],
                "workers_allocated": workers,
                "allocated_capacity": workers * capacity_per_worker,
                "produced": produced,
                "remaining": max(0.0, s["remaining"]),
                "material_ready": "YES" if ready else "NO",
            })

    results = []
    for s in states:
        completion = s["completion_day"] or horizon + 1
        tardiness = max(0, completion - s["due_day"])
        day1 = next(
            (
                r["workers_allocated"]
                for r in daily
                if r["day"] == 1 and r["order"] == s["order"]
            ),
            0,
        )
        results.append({
            "order": s["order"],
            "quantity": int(s["quantity"]),
            "due_day": s["due_day"],
            "risk": s["risk"],
            "ml_confidence": s["ml_confidence"],
            "required_daily_output": s["required_daily_output"],
            "capacity_ratio": s["capacity_ratio"],
            "day1_workers": day1,
            "completion_day": completion,
            "tardiness_days": tardiness,
            "on_time": "YES" if tardiness == 0 else "NO",
        })

    return (
        pd.DataFrame(results),
        pd.DataFrame(daily),
        {
            "available_workers": available_workers,
            "capacity_per_worker": capacity_per_worker,
            "effective_daily_capacity": available_workers * capacity_per_worker,
        },
    )

def build_recommendations(results, orders, machine_availability, labour_availability, overtime_fraction):
    recommendations = []
    order_lookup = {o["order"]: o for o in orders}

    for _, r in results.iterrows():
        order = r["order"]
        o = order_lookup[order]
        reasons = []
        actions = []

        if o["material_delay_days"] > 0:
            reasons.append(f"material unavailable for {o['material_delay_days']} working day(s)")
            actions.append("expedite material release or temporarily use the affected skilled labour on another active order")

        if machine_availability < 0.90:
            reasons.append(f"machine availability is only {machine_availability:.0%}")
            actions.append("prioritize maintenance/recovery of the bottleneck machine or protected capacity")

        if labour_availability < 0.90:
            reasons.append(f"labour availability is only {labour_availability:.0%}")
            actions.append("reassign cross-trained operators from lower-urgency orders where due-date impact remains acceptable")

        if r["capacity_ratio"] > 0.82:
            reasons.append(f"high order load relative to effective capacity ({r['capacity_ratio']:.2f})")
            actions.append("protect capacity for this order and review order sequence")

        if r["on_time"] == "NO":
            actions.append("test extra overtime, capacity sharing, subcontracting, or customer due-date renegotiation")
        elif r["risk"] == "MEDIUM":
            actions.append("monitor daily progress and trigger reallocation if actual output falls below the plan")
        else:
            actions.append("continue the current allocation while monitoring disruptions")

        if overtime_fraction > 0:
            actions.append(f"use up to {overtime_fraction:.1%} additional capacity as permitted by the model")

        recommendations.append({
            "order": order,
            "status": "LATE" if r["on_time"] == "NO" else "ON TIME",
            "risk": r["risk"],
            "reason": "; ".join(reasons) if reasons else "no major disruption indicator under the selected assumptions",
            "recommended_action": "; ".join(dict.fromkeys(actions)),
        })

    return pd.DataFrame(recommendations)
