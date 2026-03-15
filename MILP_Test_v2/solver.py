"""Stage 3 solver wrapper and result extractor."""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from pathlib import Path

import pulp


logger = logging.getLogger("solver")

_verbose = False


def set_verbose(flag: bool):
    global _verbose
    _verbose = bool(flag)


def _normalize_internal_objective(value: str, sense: int) -> float:
    parsed = float(value)
    if sense == pulp.LpMaximize:
        return -parsed
    return parsed


def _dedupe_close(values: list[float]) -> list[float]:
    deduped: list[float] = []
    for value in values:
        if not deduped or abs(deduped[-1] - value) > 1e-6:
            deduped.append(value)
    return deduped


def _parse_cbc_log(log_text: str, sense: int) -> dict:
    incumbents: list[float] = []
    bound = None
    objective = None
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
        "time_limit_hit": time_limit_hit,
        "optimal": optimal,
        "has_incumbent": objective is not None or bool(incumbents),
    }


def _parse_highs_log(log_text: str) -> dict:
    primal_pattern = re.compile(r"Primal bound\s+([+-]?\d+(?:\.\d+)?)")
    dual_pattern = re.compile(r"Dual bound\s+([+-]?\d+(?:\.\d+)?)")

    objective = None
    bound = None
    for line in log_text.splitlines():
        match = primal_pattern.search(line)
        if match:
            objective = float(match.group(1))
        match = dual_pattern.search(line)
        if match:
            bound = float(match.group(1))

    return {
        "incumbents": [objective] if objective is not None else [],
        "bound": bound,
        "objective": objective,
        "time_limit_hit": "Time limit reached" in log_text,
        "optimal": "Model status        : Optimal" in log_text or "Status            Optimal" in log_text,
        "has_incumbent": objective is not None,
    }


def _parse_solver_log(solver_name: str, log_text: str, sense: int) -> dict:
    if not log_text:
        return {
            "incumbents": [],
            "bound": None,
            "objective": None,
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
        "time_limit_hit": False,
        "optimal": False,
        "has_incumbent": False,
    }


def _make_solver(solver_name: str, d: dict, log_path: str | None) -> tuple[object, str]:
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
        return pulp.HiGHS_CMD(**kwargs), "HiGHS"

    logger.warning("Unknown solver '%s', falling back to CBC", requested)
    return pulp.PULP_CBC_CMD(**kwargs), "CBC"


def _safe_var_value(var) -> float:
    value = getattr(var, "varValue", None)
    if value is None:
        return 0.0
    return float(value)


def solve(
    d: dict,
    prob: pulp.LpProblem,
    vars: dict,
) -> dict | None:
    t_start = time.time()
    log_path = None
    log_text = ""
    selected_name = d["solver_name"]

    if not _verbose:
        temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
        log_path = temp_log.name
        temp_log.close()

    solver, solver_name = _make_solver(selected_name, d, log_path)

    try:
        status = prob.solve(solver)
    except pulp.PulpSolverError as exc:
        if solver_name == "HiGHS":
            logger.warning("HiGHS unavailable (%s), falling back to CBC", exc)
            if log_path is None:
                temp_log = tempfile.NamedTemporaryFile(delete=False, suffix=".log")
                log_path = temp_log.name
                temp_log.close()
            solver, solver_name = _make_solver("CBC", d, log_path if not _verbose else None)
            try:
                status = prob.solve(solver)
            except pulp.PulpSolverError as fallback_exc:
                logger.error("CBC solve failed: %s", fallback_exc)
                if log_path and os.path.exists(log_path):
                    os.remove(log_path)
                return None
        else:
            logger.error("Solver execution failed: %s", exc)
            if log_path and os.path.exists(log_path):
                os.remove(log_path)
            return None

    solve_time = time.time() - t_start

    if log_path and Path(log_path).exists():
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        try:
            os.remove(log_path)
        except OSError:
            pass

    parsed_log = _parse_solver_log(solver_name, log_text, prob.sense)
    for idx, incumbent in enumerate(parsed_log["incumbents"], start=1):
        logger.info("Incumbent %d: %.2f", idx, incumbent)

    sol_status = getattr(prob, "sol_status", None)
    if status == -1 or sol_status == -1:
        logger.error("Model is INFEASIBLE")
        return None
    if status == 0 and sol_status in (None, 0) and not parsed_log["has_incumbent"]:
        logger.error("Solver returned 'Not Solved'")
        return None
    if status == -2 and sol_status not in (2,):
        logger.error("Solver returned 'Unbounded'")
        return None

    if sol_status == 2 or parsed_log["time_limit_hit"]:
        status_text = "Feasible(TL)"
        logger.warning("Time limit hit - returning best feasible solution")
    elif status == 1 and sol_status in (None, 1):
        status_text = "Optimal"
        logger.info("Optimal solution found")
    elif status == 0 and parsed_log["has_incumbent"]:
        status_text = "Feasible(TL)"
        logger.warning("Time limit hit - returning best feasible solution")
    else:
        status_text = pulp.LpStatus.get(status, "Unexpected")
        if status_text == "Not Solved":
            logger.error("Solver returned 'Not Solved'")
            return None

    obj_value = pulp.value(prob.objective)
    obj_value = float(obj_value) if obj_value is not None else None

    lp_bound = None
    solver_model = getattr(getattr(prob, "solver", None), "solverModel", None)
    for candidate in (
        parsed_log.get("bound"),
        getattr(solver_model, "objBound", None) if solver_model is not None else None,
        getattr(prob, "bestBound", None),
    ):
        if candidate is not None:
            lp_bound = float(candidate)
            break
    if lp_bound is None and obj_value is not None:
        lp_bound = obj_value

    gap_pct = None
    if obj_value is not None and lp_bound is not None and abs(lp_bound) > 1e-6:
        gap_pct = abs(lp_bound - obj_value) / abs(lp_bound) * 100.0

    logger.info("Solver  : %s", solver_name)
    logger.info("Status  : %s", status_text)
    logger.info("Obj val : %s", f"{obj_value:.2f}" if obj_value is not None else "N/A")
    logger.info(
        "MIP gap : %s",
        f"{gap_pct:.4f}%"
        if gap_pct is not None
        else "N/A (not available from solver)",
    )
    logger.info("Time    : %.2fs (limit: %ss)", solve_time, d["time_limit"])

    schedule = []
    for b in d["all_batches"]:
        is_mto = d["batch_is_mto"][b]
        if is_mto:
            active = True
        else:
            active = getattr(vars["a"][b], "varValue", None) is not None and vars["a"][b].varValue >= 0.5

        if not active:
            continue

        r_assigned = None
        best_val = -1.0
        for r in d["batch_eligible_roasters"][b]:
            value = getattr(vars["x"][b][r], "varValue", None)
            if value is not None and value > best_val:
                best_val = float(value)
                r_assigned = r

        if r_assigned is None or best_val < 0.5:
            logger.warning(
                "Active batch %s has no clear roaster assignment (max x=%.3f)",
                b,
                best_val,
            )
            continue

        start_value = getattr(vars["s"][b], "varValue", None)
        if start_value is None:
            logger.warning("Active batch %s has no start time value", b)
            continue

        start = int(round(float(start_value)))
        end = start + d["process_time"]
        sku = d["batch_sku"][b]

        if not d["sku_credits_rc"][sku]:
            output_line = None
        elif r_assigned == "R3" and d["allow_r3_flex"]:
            y_val = vars["y"][b].varValue if b in vars["y"] else 0
            output_line = "L1" if y_val is not None and y_val >= 0.5 else "L2"
        elif r_assigned == "R3":
            output_line = "L2"
        else:
            output_line = d["roaster_can_output"][r_assigned][0]

        pipeline_line = d["roaster_pipeline"][r_assigned]
        pipeline_window = f"{pipeline_line}[{start}..{start + d['consume_time'] - 1}]"

        schedule.append(
            {
                "batch_id": str(b),
                "job_id": b[0] if is_mto else None,
                "sku": sku,
                "roaster": r_assigned,
                "start": int(start),
                "end": int(end),
                "pipeline": pipeline_window,
                "output_line": output_line,
                "is_mto": bool(is_mto),
            }
        )

    schedule.sort(key=lambda entry: (entry["start"], entry["roaster"], entry["batch_id"]))

    prev_sku = {r: d["roaster_initial_sku"][r] for r in d["roasters"]}
    for entry in schedule:
        roaster = entry["roaster"]
        this_sku = entry["sku"]
        if this_sku != prev_sku[roaster]:
            entry["setup"] = f"Yes({d['setup_time']}m)"
        else:
            entry["setup"] = "No"
        prev_sku[roaster] = this_sku

    psc_count = sum(1 for entry in schedule if entry["sku"] == "PSC")
    ndg_count = sum(1 for entry in schedule if entry["sku"] == "NDG")
    busta_count = sum(1 for entry in schedule if entry["sku"] == "BUSTA")

    rev_psc = float(psc_count * d["sku_revenue"]["PSC"])
    rev_ndg = float(ndg_count * d["sku_revenue"]["NDG"])
    rev_busta = float(busta_count * d["sku_revenue"]["BUSTA"])
    total_rev = rev_psc + rev_ndg + rev_busta

    tard_minutes = {}
    for job_id in d["jobs"]:
        value = getattr(vars["tard"][job_id], "varValue", None)
        tard_minutes[job_id] = max(0.0, float(value) if value is not None else 0.0)
    total_tard_min = sum(tard_minutes.values())
    total_tard_cost = total_tard_min * d["cost_tardiness"]

    total_idle_min = sum(
        _safe_var_value(vars["idle"][r][t])
        for r in d["roasters"]
        for t in vars["idle"].get(r, {})
    )
    total_idle_cost = total_idle_min * d["cost_idle"]

    total_over_min = sum(
        _safe_var_value(vars["over"][r][t])
        for r in d["roasters"]
        for t in vars["over"].get(r, {})
    )
    total_over_cost = total_over_min * d["cost_overflow"]

    total_cost = total_tard_cost + total_idle_cost + total_over_cost
    net_profit = total_rev - total_cost

    if lp_bound is None and obj_value is not None:
        lp_bound = obj_value

    if gap_pct is None and lp_bound is not None and abs(lp_bound) > 1e-6:
        gap_pct = abs(net_profit - lp_bound) / abs(lp_bound) * 100.0

    return {
        "status": status_text,
        "solve_time": round(float(solve_time), 3),
        "solver_name": solver_name,
        "obj_value": round(obj_value, 2) if obj_value is not None else None,
        "lp_bound": round(lp_bound, 2) if lp_bound is not None else None,
        "gap_pct": round(gap_pct, 4) if gap_pct is not None else None,
        "net_profit": round(net_profit, 2),
        "total_revenue": round(total_rev, 2),
        "total_costs": round(total_cost, 2),
        "psc_count": int(psc_count),
        "ndg_count": int(ndg_count),
        "busta_count": int(busta_count),
        "revenue_psc": round(rev_psc, 2),
        "revenue_ndg": round(rev_ndg, 2),
        "revenue_busta": round(rev_busta, 2),
        "tardiness_min": {job_id: round(value, 2) for job_id, value in tard_minutes.items()},
        "tard_cost": round(total_tard_cost, 2),
        "idle_min": round(total_idle_min, 2),
        "idle_cost": round(total_idle_cost, 2),
        "over_min": round(total_over_min, 2),
        "over_cost": round(total_over_cost, 2),
        "allow_r3_flex": bool(d["allow_r3_flex"]),
        "schedule": schedule,
        "sku_revenue": {sku: float(value) for sku, value in d["sku_revenue"].items()},
        "cost_tardiness": float(d["cost_tardiness"]),
        "cost_idle": float(d["cost_idle"]),
        "cost_overflow": float(d["cost_overflow"]),
    }
