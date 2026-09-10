"""Recorded daily progress and corrective actions, separate from the planner.

Targets are explicit supervisor inputs. Actual output is read only from the
existing order inputs; downtime and rework are contextual records and never
change completed quantities or the forecast. No inference is presented as a
recorded factory cause.
"""
from copy import deepcopy
from datetime import datetime, timezone
import math
from uuid import uuid4

import pandas as pd

from planner_core import validate_order_ids


ACTION_STATUSES = ("Open", "In progress", "Blocked", "Done", "Cancelled")
NOTE_COLUMNS = [
    "Order", "Process", "Daily Target", "Good Output Today", "Rework Quantity",
    "Downtime Minutes", "Recorded Reason",
]
ACTION_COLUMNS = [
    "Action ID", "Order", "Process", "Action", "Owner", "Due Date", "Status",
    "Follow-up Date", "Recorded Reason",
]


def _today(value):
    """Reject missing/invalid dates instead of silently manufacturing one."""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("Enter a valid date.")
    return parsed.date()


def _text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def _whole_number(value, label, optional=False):
    if value is None or pd.isna(value) or value == "":
        if optional:
            return None
        raise ValueError(f"{label} needs a non-negative whole number.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} needs a non-negative whole number.") from None
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError(f"{label} needs a non-negative whole number.")
    return int(number)


def _orders_by_id(orders):
    if orders.empty:
        return {}
    ids = [_text(value) for value in orders["Order"]]
    if not all(ids):
        raise ValueError("Each order needs an ID before recording production progress.")
    validate_order_ids(ids)
    return {order_id: row for order_id, (_, row) in zip(ids, orders.iterrows())}


def daily_entry_frame(orders, notes, planning_date):
    """One row per current order/date; actuals are never copied into note history."""
    day = str(_today(planning_date))
    saved = {(str(n["planning_date"]), str(n["order"])): n for n in notes}
    rows = []
    for order_id, order in _orders_by_id(orders).items():
        note = saved.get((day, order_id), {})
        rows.append({
            "Order": order_id,
            "Process": _text(order.get("Current Process", order.get("Process", ""))),
            "Daily Target": note.get("daily_target"),
            "Good Output Today": _whole_number(order.get("Actual Production Today", 0), "Actual production"),
            "Rework Quantity": note.get("rework_quantity", 0),
            "Downtime Minutes": note.get("downtime_minutes", 0),
            "Recorded Reason": note.get("recorded_reason", ""),
        })
    frame = pd.DataFrame(rows, columns=NOTE_COLUMNS)
    # Explicit numeric types let a blank target remain editable in Streamlit.
    for column in ("Daily Target", "Good Output Today", "Rework Quantity", "Downtime Minutes"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    return frame


def update_daily_notes(notes, entries, orders, planning_date, actor, now=None):
    """Validate all entries before returning a new list; never mutate the inputs."""
    day = str(_today(planning_date))
    timestamp = now or datetime.now(timezone.utc).isoformat()
    order_map = _orders_by_id(orders)
    ids = [_text(value) for value in entries["Order"]]
    validate_order_ids(ids)
    if set(ids) != set(order_map):
        raise ValueError("Production entries must match the current order list. Reopen this page after editing orders.")
    out = deepcopy(notes)
    index = {(str(n["planning_date"]), str(n["order"])): i for i, n in enumerate(out)}
    for _, entry in entries.iterrows():
        order_id = _text(entry["Order"])
        note = {
            "planning_date": day, "order": order_id,
            "process": _text(order_map[order_id].get("Current Process", order_map[order_id].get("Process", ""))),
            "daily_target": _whole_number(entry["Daily Target"], "Daily target", optional=True),
            "rework_quantity": _whole_number(entry["Rework Quantity"], "Rework quantity"),
            "downtime_minutes": _whole_number(entry["Downtime Minutes"], "Downtime minutes"),
            "recorded_reason": _text(entry["Recorded Reason"]),
        }
        key = (day, order_id)
        existing = out[index[key]] if key in index else None
        if existing and all(existing.get(k) == v for k, v in note.items()):
            continue
        # A blank untouched row is not a production observation.
        if existing is None and note["daily_target"] is None and not any(
            note[k] for k in ("rework_quantity", "downtime_minutes", "recorded_reason")
        ):
            continue
        note.update(updated_at=timestamp, updated_by=_text(actor))
        if existing is not None:
            out[index[key]] = note
        else:
            index[key] = len(out)
            out.append(note)
    return out


def daily_progress(orders, notes, planning_date):
    """Compare recorded good order output with explicit day-total targets.

    This does not infer hourly schedule adherence, root cause, or finished
    goods from process totals. Rework is neither added to nor subtracted from
    the authoritative Actual Production Today field.
    """
    frame = daily_entry_frame(orders, notes, planning_date)
    frame["Target Remaining"] = (frame["Daily Target"] - frame["Good Output Today"]).clip(lower=0)
    frame["Target Reached (%)"] = [
        round(actual / target * 100, 1) if pd.notna(target) and target > 0 else None
        for actual, target in zip(frame["Good Output Today"], frame["Daily Target"])
    ]
    frame["Progress"] = [
        "Target not set" if pd.isna(target) else
        "No output planned" if target == 0 and actual == 0 else
        "Target reached" if actual >= target else "Target remaining"
        for actual, target in zip(frame["Good Output Today"], frame["Daily Target"])
    ]
    return frame


def new_action(order, process, description, owner, due_date, follow_up_date,
               recorded_reason="", actor="", now=None, action_id=None):
    timestamp = now or datetime.now(timezone.utc).isoformat()
    if not _text(description):
        raise ValueError("Describe the action to take.")
    if not _text(owner):
        raise ValueError("Assign an owner for this action.")
    if not _text(order):
        raise ValueError("Choose an order or a factory-wide action.")
    return {
        "id": action_id or str(uuid4()), "order": _text(order),
        "process": _text(process), "description": _text(description),
        "owner": _text(owner), "due_date": str(_today(due_date)),
        "status": "Open", "follow_up_date": str(_today(follow_up_date)),
        "recorded_reason": _text(recorded_reason), "created_at": timestamp,
        "updated_at": timestamp, "updated_by": _text(actor),
    }


def action_frame(actions):
    frame = pd.DataFrame([{
        "Action ID": a["id"], "Order": a["order"], "Process": a["process"],
        "Action": a["description"], "Owner": a["owner"],
        "Due Date": _today(a["due_date"]), "Status": a["status"],
        "Follow-up Date": _today(a["follow_up_date"]),
        "Recorded Reason": a.get("recorded_reason", ""),
    } for a in actions], columns=ACTION_COLUMNS)
    return frame


def update_actions(actions, entries, actor, now=None):
    """Only update known rows; immutable IDs prevent accidental duplicate actions."""
    old = {a["id"]: a for a in actions}
    ids = [_text(value) for value in entries["Action ID"]]
    if len(set(ids)) != len(ids) or set(ids) != set(old):
        raise ValueError("The action list changed. Reload it before updating actions.")
    out = []
    timestamp = now or datetime.now(timezone.utc).isoformat()
    for _, entry in entries.iterrows():
        previous = old[_text(entry["Action ID"])]
        action = new_action(
            previous["order"], previous["process"], entry["Action"], entry["Owner"],
            entry["Due Date"], entry["Follow-up Date"], entry["Recorded Reason"],
            actor=actor, now=timestamp, action_id=previous["id"],
        )
        status = _text(entry["Status"])
        if status not in ACTION_STATUSES:
            raise ValueError("Choose a valid action status.")
        action["status"] = status
        action["created_at"] = previous["created_at"]
        tracked = ("description", "owner", "due_date", "follow_up_date", "recorded_reason", "status")
        out.append(action if any(action[k] != previous.get(k) for k in tracked) else deepcopy(previous))
    return out


def action_summary(actions, as_of):
    day = _today(as_of)
    active = [a for a in actions if a["status"] not in {"Done", "Cancelled"}]
    return {
        "open": len(active),
        "overdue": sum(_today(a["due_date"]) < day for a in active),
        "follow_up_due": sum(_today(a["follow_up_date"]) <= day for a in active),
        "completed": sum(a["status"] == "Done" for a in actions),
    }


def render_production_management(state, read_only=False, key_suffix=""):
    """Render single-column controls; caller saves state with its plan transaction.

    key_suffix should change when the caller reloads saved data or starts a day.
    Returns True when entries/actions were applied to the caller's local draft.
    """
    import streamlit as st

    notes = state.get("production_notes", [])
    actions = state.get("actions", [])
    orders = state["orders"]
    planning_date = state["planning_date"]
    actor = state.get("username", "Planner")
    changed = False
    suffix = f"{key_suffix}_{planning_date}"
    st.subheader("Daily production progress")
    st.caption("Set a daily target agreed at the start of the day. Good output comes from Production Today in the order inputs. Targets do not alter RAPID's capacity or delivery calculations.")
    try:
        progress = daily_progress(orders, notes, planning_date)
    except ValueError as exc:
        st.error(str(exc))
        return False
    if progress.empty:
        st.info("Add an order in Today's Inputs to begin recording daily progress.")
    else:
        with st.container(border=True):
            targeted = progress["Daily Target"].notna()
            st.markdown(f"**{int(targeted.sum())} of {len(progress)} orders have a daily target** · {int(progress['Good Output Today'].sum()):,} good order units recorded today")
            st.dataframe(progress[["Order", "Process", "Daily Target", "Good Output Today", "Target Remaining", "Target Reached (%)", "Progress", "Recorded Reason"]], use_container_width=True, hide_index=True)
            st.caption("Target remaining is a day-total comparison. It does not imply a delay before the shift ends. Recorded reasons are supervisor observations, not independently verified root causes.")
        if not read_only:
            with st.expander("Record targets, downtime and rework", expanded=not targeted.any()):
                st.caption("Enter good output in the existing order inputs. Rework here is context only: it does not add to or reduce completed output. Record order-level output once; do not add quantities across process stages.")
                with st.form(f"production_notes_form_{suffix}"):
                    entries = st.data_editor(
                        daily_entry_frame(orders, notes, planning_date),
                        key=f"production_notes_editor_{suffix}", use_container_width=True,
                        hide_index=True, num_rows="fixed",
                        disabled=["Order", "Process", "Good Output Today"],
                        column_config={
                            "Daily Target": st.column_config.NumberColumn(min_value=0, step=1, help="Explicit daily target, fixed independently of a revised forecast. Leave blank if not agreed."),
                            "Rework Quantity": st.column_config.NumberColumn(min_value=0, step=1),
                            "Downtime Minutes": st.column_config.NumberColumn(min_value=0, step=1, help="Downtime associated with this order. Overlapping machine downtime is not a factory total."),
                            "Recorded Reason": st.column_config.TextColumn(help="Record the observed issue and evidence. Leave blank when the cause is unknown."),
                        },
                    )
                    applied = st.form_submit_button("Apply production entries")
                if applied:
                    try:
                        updated = update_daily_notes(notes, entries, orders, planning_date, actor)
                        changed = updated != notes
                        state["production_notes"] = updated
                        st.success("Production entries applied to this draft. Save the plan to share them and keep the daily record.")
                    except ValueError as exc:
                        st.error(str(exc))

    st.subheader("Corrective actions")
    counts = action_summary(actions, planning_date)
    st.markdown(f"**{counts['open']} open** · {counts['overdue']} overdue · {counts['follow_up_due']} follow-ups due · {counts['completed']} completed")
    st.caption(f"Due status is relative to the planning date ({planning_date}). Record the owner and follow-up, then test any capacity change in Recovery Planning before applying it.")
    if actions:
        if read_only:
            st.dataframe(action_frame(actions).drop(columns="Action ID"), use_container_width=True, hide_index=True)
        else:
            with st.form(f"actions_editor_form_{suffix}"):
                action_entries = st.data_editor(
                    action_frame(actions), key=f"actions_editor_{suffix}",
                    use_container_width=True, hide_index=True, num_rows="fixed",
                    disabled=["Action ID", "Order", "Process"],
                    column_config={
                        "Action ID": None,
                        "Status": st.column_config.SelectboxColumn(options=list(ACTION_STATUSES), required=True),
                        "Due Date": st.column_config.DateColumn(required=True),
                        "Follow-up Date": st.column_config.DateColumn(required=True),
                        "Owner": st.column_config.TextColumn(required=True),
                        "Action": st.column_config.TextColumn(required=True),
                    },
                )
                update = st.form_submit_button("Apply action updates")
            if update:
                try:
                    updated = update_actions(actions, action_entries, actor)
                    changed = changed or updated != actions
                    state["actions"] = updated
                    st.success("Action updates applied to this draft. Save the plan to share them.")
                except ValueError as exc:
                    st.error(str(exc))
    else:
        st.info("No corrective actions recorded. Add an action when an order or factory constraint needs an owner.")
    if not read_only:
        with st.expander("Add a corrective action", expanded=not actions):
            order_map = _orders_by_id(orders)
            with st.form(f"new_action_form_{suffix}", clear_on_submit=True):
                selected_order = st.selectbox("Action for", ["Factory-wide"] + list(order_map), key=f"new_action_order_{suffix}")
                description = st.text_input("Action to take", placeholder="Example: inspect stitching machine and confirm repair time", key=f"new_action_description_{suffix}")
                owner = st.text_input("Responsible person", placeholder="Name or responsible team", key=f"new_action_owner_{suffix}")
                due = st.date_input("Action due date", value=_today(planning_date), key=f"new_action_due_{suffix}")
                follow_up = st.date_input("Next follow-up date", value=_today(planning_date), key=f"new_action_follow_{suffix}")
                reason = st.text_area("Observed issue or evidence", placeholder="Record what is known. Do not guess a cause.", key=f"new_action_reason_{suffix}")
                add = st.form_submit_button("Add action to draft")
            if add:
                try:
                    process = "Factory-wide" if selected_order == "Factory-wide" else order_map[selected_order].get("Current Process", order_map[selected_order].get("Process", ""))
                    action = new_action(selected_order, process, description, owner, due, follow_up, reason, actor)
                    state["actions"] = list(state.get("actions", [])) + [action]
                    changed = True
                    st.success("Action added to this draft. Save the plan to share it with the team.")
                except ValueError as exc:
                    st.error(str(exc))
    return changed
