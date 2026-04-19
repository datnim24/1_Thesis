"""Solve wrapper and result extractor for MILP_Test_v5."""

from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pulp


logger = logging.getLogger("solver")

_verbose = False
_heartbeat_interval_sec = 300.0


def set_verbose(flag: bool) -> None:
    global _verbose
    _verbose = bool(flag)


def _normalize_internal_objective(value: str, sense: int) -> float:
    parsed = float(value)
    return -parsed if sense == pulp.LpMaximize else parsed


def _dedupe_close(values: list[float]) -> list[float]:
    deduped: list[float] = []
    for value in values:
        if not deduped or abs(deduped[-1] - value) > 1e-6:
            deduped.append(value)
    return deduped


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_cbc_log(log_text: str, sense: int) -> dict[str, Any]:
    incumbents: list[float] = []
    bound = None
    objective = None
    node_count = None
    time_limit_hit = False
    optimal = False

    internal_patterns = (
        re.compile(r"Cbc0012I Integer solution of ([+-]?\d+(?:\.\d+)?)"),
        re.compile(r"Cbc0038I Solution found of ([+-]?\d+(?:\.\d+)?)"),
        re.compile(r"Cbc0038I Rounding solution of ([+-]?\d+(?:\.\d+)?)"),
        re.compile(r"Cbc0038I Mini branch and bound improved solution from [^ ]+ to ([+-]?\d+(?:\.\d+)?)"),
    )
    partial_pattern = re.compile(
        r"Cbc0005I Partial search - best objective ([+-]?\d+(?:\.\d+)?) "
        r"\(best possible ([+-]?\d+(?:\.\d+)?)\)"
    )
    complete_pattern = re.compile(r"Cbc0001I Search completed - best objective ([+-]?\d+(?:\.\d+)?)")
    objective_pattern = re.compile(r"Objective value:\s+([+-]?\d+(?:\.\d+)?)")
    upper_bound_pattern = re.compile(r"Upper bound:\s+([+-]?\d+(?:\.\d+)?)")
    enumerated_nodes_pattern = re.compile(r"Enumerated nodes:\s+(\d+)")
    partial_nodes_pattern = re.compile(r"(\d+)\s+nodes")

    for line in log_text.splitlines():
        for pattern in internal_patterns:
            match = pattern.search(line)
            if match:
                incumbents.append(_normalize_internal_objective(match.group(1), sense))
                break

        match = partial_pattern.search(line)
        if match:
            objective = _normalize_internal_objective(match.group(1), sense)
            bound = _normalize_internal_objective(match.group(2), sense)

        match = complete_pattern.search(line)
        if match and objective is None:
            objective = _normalize_internal_objective(match.group(1), sense)

        match = objective_pattern.search(line)
        if match:
            objective = float(match.group(1))

        match = upper_bound_pattern.search(line)
        if match:
            bound = float(match.group(1))

        match = enumerated_nodes_pattern.search(line)
        if match:
            node_count = int(match.group(1))
        elif "nodes" in line.lower():
            match = partial_nodes_pattern.search(line)
            if match:
                node_count = int(match.group(1))

        if "Stopped on time limit" in line or "Exiting on maximum time" in line:
            time_limit_hit = True
        if "Optimal solution found" in line:
            optimal = True

    incumbents = _dedupe_close(incumbents)
    if objective is None and incumbents:
        objective = incumbents[-1]

    return {
        "incumbents": incumbents,
        "bound": bound,
        "objective": objective,
        "node_count": node_count,
        "time_limit_hit": time_limit_hit,
        "optimal": optimal,
        "has_incumbent": objective is not None or bool(incumbents),
    }


def _parse_highs_log(log_text: str) -> dict[str, Any]:
    primal_pattern = re.compile(r"Primal bound\s+([+-]?\d+(?:\.\d+)?)")
    dual_pattern = re.compile(r"Dual bound\s+([+-]?\d+(?:\.\d+)?)")
    nodes_pattern = re.compile(r"Nodes\s+(\d+)")

    objective = None
    bound = None
    node_count = None
    for line in log_text.splitlines():
        match = primal_pattern.search(line)
        if match:
            objective = float(match.group(1))
        match = dual_pattern.search(line)
        if match:
            bound = float(match.group(1))
        match = nodes_pattern.search(line)
        if match:
            node_count = int(match.group(1))

    return {
        "incumbents": [objective] if objective is not None else [],
        "bound": bound,
        "objective": objective,
        "node_count": node_count,
        "time_limit_hit": "Time limit reached" in log_text,
        "optimal": "Model status        : Optimal" in log_text or "Status            Optimal" in log_text,
        "has_incumbent": objective is not None,
    }


def _parse_solver_log(solver_name: str, log_text: str, sense: int) -> dict[str, Any]:
    if not log_text:
        return {
            "incumbents": [],
            "bound": None,
            "objective": None,
            "node_count": None,
            "time_limit_hit": False,
            "optimal": False,
            "has_incumbent": False,
        }
    if solver_name == "CBC":
        return _parse_cbc_log(log_text, sense)
    if solver_name == "HiGHS":
        return _parse_highs_log(log_text)
    return {
        "incumbents": [],
        "bound": None,
        "objective": None,
        "node_count": None,
        "time_limit_hit": False,
        "optimal": False,
        "has_incumbent": False,
    }


def _extract_highs_info(prob: pulp.LpProblem) -> dict[str, Any]:
    solver_model = getattr(prob, "solverModel", None)
    if solver_model is None or not hasattr(solver_model, "getInfo"):
        return {
            "objective": None,
            "bound": None,
            "node_count": None,
            "gap_pct": None,
        }

    try:
        info = solver_model.getInfo()
    except Exception:
        return {
            "objective": None,
            "bound": None,
            "node_count": None,
            "gap_pct": None,
        }

    objective = _finite_float(getattr(info, "objective_function_value", None))
    bound = _finite_float(getattr(info, "mip_dual_bound", None))

    node_count = None
    raw_nodes = getattr(info, "mip_node_count", None)
    if raw_nodes is not None:
        try:
            node_count = int(float(raw_nodes))
        except (TypeError, ValueError):
            node_count = None

    gap_pct = None
    if objective is not None and bound is not None and abs(bound) > 1e-6:
        gap_pct = abs(bound - objective) / abs(bound) * 100.0
    else:
        raw_gap = _finite_float(getattr(info, "mip_gap", None))
        if raw_gap is not None:
            gap_pct = raw_gap * 100.0 if raw_gap <= 1.0 else raw_gap

    return {
        "objective": objective,
        "bound": bound,
        "node_count": node_count,
        "gap_pct": gap_pct,
    }


def _extract_log_progress(log_text: str) -> dict[str, Any]:
    live_obj = None
    live_bound = None
    live_nodes = None

    for line in reversed(log_text.splitlines()):
        if live_obj is None:
            match = re.search(r"Primal bound\s+([+-]?\d+(?:\.\d+)?)", line)
            if match:
                live_obj = float(match.group(1))
        if live_bound is None:
            match = re.search(r"Dual bound\s+([+-]?\d+(?:\.\d+)?)", line)
            if match:
                live_bound = float(match.group(1))
        if live_nodes is None:
            match = re.search(r"Nodes\s+(\d+)", line)
            if match:
                live_nodes = int(match.group(1))
        if live_obj is not None and live_bound is not None and live_nodes is not None:
            break

    if live_obj is None:
        for line in reversed(log_text.splitlines()):
            match = re.search(r"Cbc0012I Integer solution of ([+-]?\d+(?:\.\d+)?)", line)
            if not match:
                match = re.search(r"Cbc0038I Solution found of ([+-]?\d+(?:\.\d+)?)", line)
            if match:
                live_obj = -float(match.group(1))
                break

    if live_nodes is None:
        for line in reversed(log_text.splitlines()):
            match = re.search(r"Enumerated nodes:\s+(\d+)", line)
            if not match and "nodes" in line.lower():
                match = re.search(r"(\d+)\s+nodes", line)
            if match:
                live_nodes = int(match.group(1))
                break

    live_gap = None
    if live_obj is not None and live_bound is not None and abs(live_bound) > 1e-6:
        live_gap = abs(live_bound - live_obj) / abs(live_bound) * 100.0

    return {
        "objective": live_obj,
        "bound": live_bound,
        "node_count": live_nodes,
        "gap_pct": live_gap,
    }


def _make_solver(solver_name: str, d: dict[str, Any], log_path: str | None) -> tuple[object, str]:
    requested = (solver_name or "").strip()
    kwargs = {
        "timeLimit": d["time_limit"],
        "gapRel": d["mip_gap"],
        "msg": _verbose,
    }
    if log_path is not None:
        kwargs["logPath"] = log_path

    if requested == "CBC":
        return pulp.PULP_CBC_CMD(**kwargs), "CBC"
    if requested == "HiGHS":
        highs_kwargs = {
            "timeLimit": d["time_limit"],
            "gapRel": d["mip_gap"],
            "msg": False,
            "presolve": d.get("presolve_mode", "off"),
            "mip_detect_symmetry": False,
        }
        if hasattr(pulp, "HiGHS"):
            try:
                solver = pulp.HiGHS(**highs_kwargs)
                if solver.available():
                    return solver, "HiGHS"
            except Exception:
                logger.debug("HiGHS API unavailable, falling back to HiGHS_CMD", exc_info=True)
        return pulp.HiGHS_CMD(**kwargs), "HiGHS"

    logger.warning("Unknown solver '%s', falling back to CBC", requested)
    return pulp.PULP_CBC_CMD(**kwargs), "CBC"


def _solver_heartbeat(
    stop_event: threading.Event,
    start_time: float,
    time_limit: float | int | None,
    log_path: str | None,
    prob: pulp.LpProblem,
) -> None:
    while not stop_event.wait(_heartbeat_interval_sec):
        elapsed = time.time() - start_time
        remaining = max(0.0, float(time_limit) - elapsed) if time_limit else None
        live_metrics = _extract_highs_info(prob)
        if (
            live_metrics["objective"] is None
            and live_metrics["bound"] is None
            and live_metrics["node_count"] is None
            and log_path
            and Path(log_path).exists()
        ):
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                live_metrics = _extract_log_progress(text)
            except OSError:
                pass

        parts = [f"elapsed={elapsed:.0f}s"]
        if remaining is not None:
            parts.append(f"remaining={remaining:.0f}s")
        if live_metrics["objective"] is not None:
            parts.append(f"incumbent={live_metrics['objective']:,.0f}")
        if live_metrics["bound"] is not None:
            parts.append(f"bound={live_metrics['bound']:,.0f}")
        if live_metrics["gap_pct"] is not None:
            parts.append(f"gap={live_metrics['gap_pct']:.2f}%")
        if live_metrics["node_count"] is not None:
            parts.append(f"nodes={live_metrics['node_count']:,}")
        logger.info("Solve progress | %s", " | ".join(parts))


def _selected_batch_start(vars_dict: dict[str, Any], batch_id: Any) -> tuple[str | None, int | None, float]:
    best_roaster = None
    best_start = None
    best_value = -1.0
    for roaster, start_map in vars_dict.get("x_ti", {}).get(batch_id, {}).items():
        for start, xti_var in start_map.items():
            value = getattr(xti_var, "varValue", None)
            if value is not None and float(value) > best_value:
                best_value = float(value)
                best_roaster = roaster
                best_start = int(start)
    return best_roaster, best_start, best_value


def _safe_var_value(var: Any) -> float:
    value = getattr(var, "varValue", None)
    return 0.0 if value is None else float(value)


def solve(d: dict[str, Any], prob: pulp.LpProblem, vars_dict: dict[str, Any]) -> dict[str, Any] | None:
    t_start = time.time()
    log_path = None
    log_text = ""
    selected_name = d["solver_name"]

    build_meta = getattr(prob, "_build_meta", None)
    if build_meta:
        logger.info("-" * 60)
        logger.info("MODEL STATISTICS (v5)")
        logger.info("  Formulation : %s", build_meta.get("formulation", "unknown"))
        logger.info("  Build time  : %.2fs", build_meta.get("build_seconds", 0.0))
        logger.info("  Variables   : %d", build_meta.get("total_vars", 0))
        logger.info("  Constraints : %d", build_meta.get("total_constraints", 0))
        logger.info("  Notes       : %s", "; ".join(build_meta.get("notes", [])))
        logger.info("-" * 60)

    temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
    log_path = temp_log.name
    temp_log.close()

    solver, solver_name = _make_solver(selected_name, d, log_path)
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_solver_heartbeat,
        args=(heartbeat_stop, t_start, d.get("time_limit"), log_path, prob),
        daemon=True,
        name="solver-heartbeat",
    )
    heartbeat_thread.start()

    try:
        status = prob.solve(solver)
    except pulp.PulpSolverError as exc:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)
        if solver_name == "HiGHS":
            logger.warning("HiGHS unavailable (%s), falling back to CBC", exc)
            if log_path is None:
                temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
                log_path = temp_log.name
                temp_log.close()
            solver, solver_name = _make_solver("CBC", d, log_path)
            heartbeat_stop = threading.Event()
            heartbeat_thread = threading.Thread(
                target=_solver_heartbeat,
                args=(heartbeat_stop, t_start, d.get("time_limit"), log_path, prob),
                daemon=True,
                name="solver-heartbeat",
            )
            heartbeat_thread.start()
            try:
                status = prob.solve(solver)
            except pulp.PulpSolverError as fallback_exc:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
                logger.error("CBC solve failed: %s", fallback_exc)
                if log_path and os.path.exists(log_path):
                    os.remove(log_path)
                return None
        else:
            logger.error("Solver execution failed: %s", exc)
            if log_path and os.path.exists(log_path):
                os.remove(log_path)
            return None
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1.0)

    solve_time = time.time() - t_start

    if log_path and Path(log_path).exists():
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        try:
            os.remove(log_path)
        except OSError:
            pass

    parsed_log = _parse_solver_log(solver_name, log_text, prob.sense)
    sol_status = getattr(prob, "sol_status", None)
    if status == -1 or sol_status == -1:
        logger.error("Model is infeasible")
        return None
    if status == 0 and sol_status in (None, 0) and not parsed_log["has_incumbent"]:
        logger.error("Solver returned 'Not Solved'")
        return None
    if status == -2 and sol_status not in (2,):
        logger.error("Solver returned 'Unbounded'")
        return None

    if sol_status == 2 or parsed_log["time_limit_hit"]:
        status_text = "Feasible(TL)"
    elif status == 1 and sol_status in (None, 1):
        status_text = "Optimal"
    elif status == 0 and parsed_log["has_incumbent"]:
        status_text = "Feasible(TL)"
    else:
        status_text = pulp.LpStatus.get(status, "Unexpected")
        if status_text == "Not Solved":
            return None

    obj_value = pulp.value(prob.objective)
    obj_value = float(obj_value) if obj_value is not None else None

    highs_info = _extract_highs_info(prob)
    lp_bound = None
    solver_model = getattr(prob, "solverModel", None)
    for candidate in (
        highs_info.get("bound"),
        parsed_log.get("bound"),
        getattr(solver_model, "objBound", None) if solver_model is not None else None,
        getattr(prob, "bestBound", None),
    ):
        if candidate is not None:
            lp_bound = float(candidate)
            break
    if lp_bound is None and obj_value is not None and status_text == "Optimal":
        lp_bound = obj_value

    gap_pct = None
    if obj_value is not None and lp_bound is not None and abs(lp_bound) > 1e-6:
        gap_pct = abs(lp_bound - obj_value) / abs(lp_bound) * 100.0
    elif highs_info.get("gap_pct") is not None:
        gap_pct = float(highs_info["gap_pct"])

    node_count = None
    for candidate in (
        highs_info.get("node_count"),
        parsed_log.get("node_count"),
    ):
        if candidate is not None:
            node_count = int(candidate)
            break

    logger.info("-" * 60)
    logger.info("SOLVE RESULT")
    logger.info("  Solver    : %s", solver_name)
    logger.info("  Status    : %s", status_text)
    logger.info("  Objective : %s", f"${obj_value:,.2f}" if obj_value is not None else "N/A")
    logger.info("  LP bound  : %s", f"${lp_bound:,.2f}" if lp_bound is not None else "N/A")
    logger.info("  MIP gap   : %s", f"{gap_pct:.4f}%" if gap_pct is not None else "N/A")
    logger.info("  Nodes     : %s", f"{node_count:,}" if node_count is not None else "N/A")
    logger.info("  Solve time: %.2fs (limit=%ss)", solve_time, d["time_limit"])
    logger.info("-" * 60)

    schedule: list[dict[str, Any]] = []
    for batch_id in d["all_batches"]:
        is_mto = d["batch_is_mto"][batch_id]
        active = True if is_mto else _safe_var_value(vars_dict["a"][batch_id]) >= 0.5
        if not active:
            continue
        roaster, start, best_val = _selected_batch_start(vars_dict, batch_id)
        if roaster is None or start is None or best_val < 0.5:
            logger.warning("Active batch %s has no selected start (max=%.3f)", batch_id, best_val)
            continue
        sku = d["batch_sku"][batch_id]
        duration = d["roast_time_by_sku"][sku]
        output_line = None
        if d["sku_credits_rc"][sku]:
            if roaster == "R3" and d["allow_r3_flex"]:
                z_map = vars_dict.get("z_route", {}).get(batch_id, {})
                z_l1 = _safe_var_value(z_map.get("L1", {}).get(start))
                z_l2 = _safe_var_value(z_map.get("L2", {}).get(start))
                output_line = "L1" if z_l1 >= z_l2 else "L2"
            else:
                output_line = d["roaster_can_output"][roaster][0]

        schedule.append(
            {
                "batch_id": str(batch_id),
                "job_id": batch_id[0] if is_mto else None,
                "sku": sku,
                "roaster": roaster,
                "start": int(start),
                "end": int(start + duration),
                "pipeline": d["roaster_pipeline"][roaster],
                "pipeline_start": int(start),
                "pipeline_end": int(start + d["consume_time"]),
                "output_line": output_line,
                "is_mto": bool(is_mto),
                "status": "completed",
            }
        )

    schedule.sort(key=lambda entry: (entry["start"], entry["roaster"], entry["batch_id"]))

    prev_sku = {roaster: d["roaster_initial_sku"][roaster] for roaster in d["roasters"]}
    setup_events = 0
    for entry in schedule:
        roaster = entry["roaster"]
        if entry["sku"] != prev_sku[roaster]:
            entry["setup"] = f"Yes({d['setup_time']}m)"
            setup_events += 1
        else:
            entry["setup"] = "No"
        prev_sku[roaster] = entry["sku"]

    restocks: list[dict[str, Any]] = []
    for pair, start_map in vars_dict.get("restock", {}).items():
        for start, rst_var in start_map.items():
            if _safe_var_value(rst_var) >= 0.5:
                restocks.append(
                    {
                        "line_id": pair[0],
                        "sku": pair[1],
                        "start": int(start),
                        "end": int(start + d["restock_duration"]),
                        "qty": int(d["restock_qty"]),
                    }
                )
    restocks.sort(key=lambda item: (item["start"], item["line_id"], item["sku"]))

    psc_count = sum(1 for entry in schedule if entry["sku"] == "PSC")
    ndg_count = sum(1 for entry in schedule if entry["sku"] == "NDG")
    busta_count = sum(1 for entry in schedule if entry["sku"] == "BUSTA")

    revenue_psc = psc_count * d["sku_revenue"]["PSC"]
    revenue_ndg = ndg_count * d["sku_revenue"]["NDG"]
    revenue_busta = busta_count * d["sku_revenue"]["BUSTA"]
    total_revenue = revenue_psc + revenue_ndg + revenue_busta

    tardiness_min = {
        job_id: max(0.0, _safe_var_value(vars_dict["tard"][job_id]))
        for job_id in d["jobs"]
    }
    tard_cost = sum(tardiness_min.values()) * d["cost_tardiness"]

    idle_min = sum(
        _safe_var_value(vars_dict["idle"][roaster][slot])
        for roaster in d["roasters"]
        for slot in vars_dict["idle"].get(roaster, {})
    )
    over_min = sum(
        _safe_var_value(vars_dict["over"][roaster][slot])
        for roaster in d["roasters"]
        for slot in vars_dict["over"].get(roaster, {})
    )
    idle_cost = idle_min * d["cost_idle"]
    over_cost = over_min * d["cost_overflow"]
    setup_cost = setup_events * d["cost_setup"]

    total_costs = tard_cost + setup_cost + idle_cost + over_cost
    net_profit = total_revenue - total_costs

    gc_final = {
        f"{pair[0]}_{pair[1]}": int(round(_safe_var_value(slot_map[d['shift_length'] - 1])))
        for pair, slot_map in vars_dict.get("gc", {}).items()
    }
    rc_final = {
        line_id: int(round(_safe_var_value(vars_dict["rc"][line_id][d["shift_length"] - 1])))
        for line_id in d["lines"]
    }

    return {
        "status": status_text,
        "solve_time": round(float(solve_time), 3),
        "solver_name": solver_name,
        "obj_value": round(obj_value, 2) if obj_value is not None else None,
        "lp_bound": round(lp_bound, 2) if lp_bound is not None else None,
        "gap_pct": round(gap_pct, 4) if gap_pct is not None else None,
        "node_count": int(node_count) if node_count is not None else None,
        "net_profit": round(net_profit, 2),
        "objective_profit": round(obj_value, 2) if obj_value is not None else None,
        "total_revenue": round(total_revenue, 2),
        "total_costs": round(total_costs, 2),
        "psc_count": int(psc_count),
        "ndg_count": int(ndg_count),
        "busta_count": int(busta_count),
        "total_batches": int(psc_count + ndg_count + busta_count),
        "revenue_psc": round(revenue_psc, 2),
        "revenue_ndg": round(revenue_ndg, 2),
        "revenue_busta": round(revenue_busta, 2),
        "tardiness_min": {job_id: round(value, 2) for job_id, value in tardiness_min.items()},
        "tard_cost": round(tard_cost, 2),
        "setup_events": int(setup_events),
        "setup_cost": round(setup_cost, 2),
        "idle_min": round(idle_min, 2),
        "idle_cost": round(idle_cost, 2),
        "over_min": round(over_min, 2),
        "over_cost": round(over_cost, 2),
        "restock_count": int(len(restocks)),
        "allow_r3_flex": bool(d["allow_r3_flex"]),
        "inventory_option": "FULL_DETERMINISTIC",
        "schedule": schedule,
        "restocks": restocks,
        "sku_revenue": {sku: float(value) for sku, value in d["sku_revenue"].items()},
        "cost_tardiness": float(d["cost_tardiness"]),
        "cost_idle": float(d["cost_idle"]),
        "cost_overflow": float(d["cost_overflow"]),
        "cost_setup": float(d["cost_setup"]),
        "roast_time_by_sku": dict(d["roast_time_by_sku"]),
        "gc_init": {f"{pair[0]}_{pair[1]}": int(value) for pair, value in d["gc_init"].items()},
        "gc_final": gc_final,
        "gc_capacity": {f"{pair[0]}_{pair[1]}": int(value) for pair, value in d["gc_capacity"].items()},
        "rc_init": dict(d["rc_init"]),
        "rc_final": rc_final,
        "restock_duration": int(d["restock_duration"]),
        "restock_qty": int(d["restock_qty"]),
        "input_dir": d.get("input_dir"),
        "model_notes": list((build_meta or {}).get("notes", [])),
    }
