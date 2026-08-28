import math
import pandas as pd

RISK_MULTIPLIER = {"LOW": 1.00, "MEDIUM": 1.18, "HIGH": 1.40}


def business_days_between(start_date, due_date):
    """Business days from planning date through due date (minimum 1)."""
    start = pd.Timestamp(start_date).normalize()
    due = pd.Timestamp(due_date).normalize()
    if due < start:
        return 1
    return max(1, len(pd.bdate_range(start, due)))


def business_days_until_ready(start_date, ready_date):
    """Business days before material becomes available; 0 means available today."""
    start = pd.Timestamp(start_date).normalize()
    ready = pd.Timestamp(ready_date).normalize()
    if ready <= start:
        return 0
    # Count business days [start, ready), so a Friday-ready material is usable Friday.
    return len(pd.bdate_range(start, ready - pd.Timedelta(days=1)))


def add_business_days(start_date, working_days):
    d = pd.Timestamp(start_date).normalize()
    if working_days <= 0:
        return d
    return pd.bdate_range(d, periods=working_days + 1)[-1]


def recommend_overtime(
    orders,
    base_capacity,
    aggregate_machine_availability,
    labour_availability,
    max_overtime,
):
    """Recommend an overtime buffer from due-date workload pressure, not a naive sum of all order run-rates.

    The peak cumulative output required by each due-date milestone is compared with the
    regular effective capacity. Material delay is also checked at the individual-order
    level so a late material release can trigger a recovery buffer.
    """
    active = [o for o in orders if float(o.get("quantity", 0)) > 0]
    if not active:
        return 0.0

    milestone_required = 0.0
    due_days = sorted({max(1, int(o.get("days_remaining", 1))) for o in active})
    for d in due_days:
        cumulative_qty = sum(
            float(o.get("quantity", 0))
            for o in active
            if max(1, int(o.get("days_remaining", 1))) <= d
        )
        milestone_required = max(milestone_required, cumulative_qty / max(1, d))

    individual_after_material = max(
        float(o.get("quantity", 0))
        / max(1, int(o.get("days_remaining", 1)) - int(o.get("material_delay_days", 0)))
        for o in active
    )
    required_capacity = max(milestone_required, individual_after_material)

    regular_capacity = (
        float(base_capacity)
        * max(0.0, float(aggregate_machine_availability))
        * max(0.0, float(labour_availability))
    )
    if regular_capacity <= 1e-9:
        return float(max_overtime)
    needed = max(0.0, required_capacity / regular_capacity - 1.0)
    return min(float(max_overtime), needed)

def _capacity_per_worker(state, base_capacity, benchmark_workers, overtime_fraction):
    benchmark_workers = max(1, int(benchmark_workers))
    machine_factor = max(0.0, min(1.0, float(state.get("machine_availability", 1.0))))
    return (
        float(base_capacity)
        / benchmark_workers
        * machine_factor
        * (1.0 + max(0.0, float(overtime_fraction)))
    )


def allocate_workers_for_day(
    states,
    available_workers,
    base_capacity,
    benchmark_workers,
    overtime_fraction,
    day,
):
    """
    Allocate the shared workforce among material-ready orders using earliest-due-date
    protection, adjusted for ML risk and each order's current-process machine factor.
    """
    ready = [
        s for s in states
        if s["remaining"] > 1e-9 and day >= int(s.get("material_delay_days", 0))
    ]
    allocation = {s["order"]: 0 for s in states}
    if not ready or available_workers <= 0:
        return allocation

    scored = []
    for s in ready:
        days_left = max(1, int(s["due_day"]) - day)
        cpw = max(
            1e-9,
            _capacity_per_worker(s, base_capacity, benchmark_workers, overtime_fraction),
        )
        required_workers = s["remaining"] / days_left / cpw
        risk_weight = RISK_MULTIPLIER.get(str(s.get("risk", "LOW")).upper(), 1.0)
        priority = risk_weight * (1.0 + 1.0 / days_left)
        scored.append((s, required_workers, priority, days_left))

    # Earliest due date first. For the same due pressure, protect higher-risk work.
    scored.sort(key=lambda x: (x[3], -x[2]))
    remaining_workers = int(available_workers)

    # Give each order the approximate workforce needed to maintain its due-date run rate,
    # subject to the shared workforce ceiling. This lets later-due orders defer labour when
    # an earlier order needs temporary protection.
    for s, required_workers, _, _ in scored:
        if remaining_workers <= 0:
            break
        target = max(1, int(math.ceil(required_workers)))
        give = min(target, remaining_workers)
        allocation[s["order"]] += give
        remaining_workers -= give

    # Any surplus workforce goes to the largest risk-adjusted remaining production gap.
    while remaining_workers > 0 and scored:
        best_state = None
        best_score = -1e18
        for s, _, priority, days_left in scored:
            allocated = allocation[s["order"]]
            cpw = max(1e-9, _capacity_per_worker(s, base_capacity, benchmark_workers, overtime_fraction))
            planned_output = allocated * cpw
            daily_need = s["remaining"] / max(1, days_left)
            shortfall = daily_need - planned_output
            score = shortfall * priority
            if score > best_score:
                best_score = score
                best_state = s
        allocation[best_state["order"]] += 1
        remaining_workers -= 1

    return allocation

def simulate_orders(
    orders,
    base_capacity,
    benchmark_workers,
    labour_availability,
    overtime_fraction,
    horizon=120,
):
    """
    Simulate remaining order completion with shared labour and order/process-specific
    machine availability.
    """
    benchmark_workers = max(1, int(benchmark_workers))
    # Round rather than floor so an exact attendance ratio (e.g. 110/150)
    # returns the actual integer attendance instead of occasionally losing one
    # worker because of floating-point representation (109.999999...).
    available_workers = max(
        0,
        int(round(benchmark_workers * max(0.0, min(1.0, float(labour_availability))))),
    )

    states = []
    for o in orders:
        states.append({
            **o,
            "remaining": max(0.0, float(o.get("quantity", 0))),
            "completion_day": None,
        })

    daily = []
    for day in range(int(horizon)):
        if all(s["remaining"] <= 1e-9 for s in states):
            break

        allocation = allocate_workers_for_day(
            states,
            available_workers,
            base_capacity,
            benchmark_workers,
            overtime_fraction,
            day,
        )

        for s in states:
            ready = day >= int(s.get("material_delay_days", 0))
            workers = int(allocation.get(s["order"], 0))
            cpw = _capacity_per_worker(
                s, base_capacity, benchmark_workers, overtime_fraction
            )
            produced = 0.0

            if ready and workers > 0 and s["remaining"] > 0:
                produced = min(s["remaining"], workers * cpw)
                s["remaining"] -= produced
                if s["remaining"] <= 1e-9 and s["completion_day"] is None:
                    s["completion_day"] = day + 1

            daily.append({
                "day": day + 1,
                "order": s["order"],
                "current_process": s.get("current_process", ""),
                "risk": s.get("risk", "LOW"),
                "workers_allocated": workers,
                "allocated_capacity": workers * cpw,
                "produced": produced,
                "remaining": max(0.0, s["remaining"]),
                "material_ready": "YES" if ready else "NO",
                "machine_availability": float(s.get("machine_availability", 1.0)),
            })

    results = []
    for s in states:
        completion = s["completion_day"] or int(horizon) + 1
        projected_delay = max(0, completion - int(s["due_day"]))
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
            "original_quantity": int(round(float(s.get("original_quantity", s.get("quantity", 0))))),
            "quantity": int(round(float(s.get("quantity", 0)))),
            "due_day": int(s["due_day"]),
            "current_process": s.get("current_process", ""),
            "material_status": s.get("material_status", "Ready"),
            "risk": s.get("risk", "LOW"),
            "ml_confidence": float(s.get("ml_confidence", 0)),
            "required_daily_output": float(s.get("required_daily_output", 0)),
            "capacity_ratio": float(s.get("capacity_ratio", 0)),
            "machine_availability": float(s.get("machine_availability", 1.0)),
            "day1_workers": int(day1),
            "completion_day": int(completion),
            "projected_delay_days": int(projected_delay),
            "on_time": "YES" if projected_delay == 0 else "NO",
        })

    return (
        pd.DataFrame(results),
        pd.DataFrame(daily),
        {
            "available_workers": int(available_workers),
            "base_capacity_per_worker": float(base_capacity) / benchmark_workers,
        },
    )


def build_recommendations(results, orders, labour_availability, overtime_fraction):
    """Build concise order-level reasons/actions used by the management decision centre."""
    recommendations = []
    order_lookup = {o["order"]: o for o in orders}

    for _, r in results.iterrows():
        order = r["order"]
        o = order_lookup[order]
        reasons = []
        actions = []

        status = str(o.get("material_status", "Ready"))
        if status not in {"Ready", "Partially Ready"}:
            reasons.append(f"material status is {status}")
            actions.append("expedite material clearance/replacement and keep the order out of constrained production allocation until material is released")
        elif status == "Partially Ready":
            reasons.append("material is only partially ready")
            actions.append("confirm component readiness before releasing the next production stage")

        machine = float(o.get("machine_availability", 1.0))
        process = str(o.get("current_process", "current process"))
        if machine < 0.90:
            reasons.append(f"{process} machine availability is {machine:.0%}")
            actions.append(f"prioritize maintenance/recovery at {process} and protect capacity for urgent ready orders")

        if labour_availability < 0.90:
            reasons.append(f"labour availability is {labour_availability:.0%}")
            actions.append("reassign eligible/cross-trained operators from lower-urgency work where due-date impact remains acceptable")

        if float(r.get("capacity_ratio", 0)) >= 1.0:
            reasons.append(f"required output exceeds the current capacity basis ({float(r['capacity_ratio']):.2f})")
            actions.append("protect capacity for this order and test overtime/resource recovery before committing the schedule")
        elif float(r.get("capacity_ratio", 0)) >= 0.85:
            reasons.append(f"capacity is tight ({float(r['capacity_ratio']):.2f})")
            actions.append("monitor actual output daily and prepare a recovery action if the capacity buffer falls")

        if r["on_time"] == "NO":
            actions.append("test overtime, machine recovery, capacity sharing/subcontracting, or due-date renegotiation")
        elif str(r.get("risk", "LOW")).upper() == "MEDIUM":
            actions.append("monitor daily progress and trigger reallocation if actual output falls below plan")
        elif not actions:
            actions.append("continue the current plan while monitoring labour, machine and material changes")

        if overtime_fraction > 0:
            actions.append(f"use up to {overtime_fraction:.1%} additional capacity as recommended within the configured limit")

        recommendations.append({
            "order": order,
            "status": "PROJECTED LATE" if r["on_time"] == "NO" else "ON TIME",
            "risk": str(r.get("risk", "LOW")).upper(),
            "reason": "; ".join(dict.fromkeys(reasons)) if reasons else "no major disruption indicator under the selected assumptions",
            "recommended_action": "; ".join(dict.fromkeys(actions)),
        })

    return pd.DataFrame(recommendations)
