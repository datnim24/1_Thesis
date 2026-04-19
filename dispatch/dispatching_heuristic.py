"""Urgency-first dispatching heuristic with GC-aware restock support.

Handles both roasting decisions (per-roaster) and restock decisions (global).
Consumes env state and params but lives outside env/.
"""

from __future__ import annotations

import math
from typing import Optional

from env.simulation_engine import SimulationEngine


class DispatchingHeuristic:
    """MTO-urgency dispatching + reorder-point GC restock."""

    MTO_URGENCY_THRESHOLD = 0.7

    def __init__(self, params: dict):
        self.params = params
        self._roast_time = dict(params["roast_time_by_sku"])
        self._engine: SimulationEngine | None = None
        self.max_depletion_rate: dict[tuple[str, str], float] = {}
        self.eligible_roaster_count: dict[tuple[str, str], int] = {}
        self.reorder_point_exact: dict[tuple[str, str], int] = {}
        self.reorder_point_impl: dict[tuple[str, str], int] = {}
        self._build_reorder_points()

    def _get_engine(self) -> SimulationEngine:
        if self._engine is None:
            self._engine = SimulationEngine(self.params)
        return self._engine

    def _build_reorder_points(self) -> None:
        lead_time = max(0, int(self.params.get("restock_duration", 0)))
        for raw_pair in self.params.get("feasible_gc_pairs", []):
            line_id, sku = tuple(raw_pair)
            pair = (line_id, sku)
            eligible_count = sum(
                1
                for roaster_id in self.params["roasters"]
                if self.params["R_pipe"][roaster_id] == line_id
                and sku in self.params["R_elig_skus"][roaster_id]
            )
            roast_time = max(1, int(self._roast_time[sku]))
            depletion_rate = eligible_count / roast_time
            self.eligible_roaster_count[pair] = eligible_count
            self.max_depletion_rate[pair] = depletion_rate
            self.reorder_point_exact[pair] = int(math.ceil(lead_time * depletion_rate))
            self.reorder_point_impl[pair] = int(math.ceil((lead_time + 1) * depletion_rate))

    def get_reorder_point_report(self) -> dict[str, dict[str, float | int | str]]:
        report: dict[str, dict[str, float | int | str]] = {}
        for raw_pair in self.params.get("feasible_gc_pairs", []):
            line_id, sku = tuple(raw_pair)
            pair = (line_id, sku)
            report[f"{line_id}_{sku}"] = {
                "line": line_id,
                "sku": sku,
                "roast_time": int(self._roast_time[sku]),
                "eligible_roaster_count": int(self.eligible_roaster_count[pair]),
                "restock_duration": int(self.params.get("restock_duration", 0)),
                "max_depletion_rate": float(self.max_depletion_rate[pair]),
                "ROP_exact": int(self.reorder_point_exact[pair]),
                "ROP_impl": int(self.reorder_point_impl[pair]),
            }
        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _action_requires_setup(self, state, roaster_id: str, sku: str) -> bool:
        return state.last_sku.get(roaster_id) != sku

    def _would_overlap_downtime(self, roaster_id: str, start: int, duration: int) -> bool:
        for slot in range(start, min(start + duration, self.params["SL"])):
            if slot in self.params["downtime_slots"].get(roaster_id, set()):
                return True
        return False

    def _can_take_action(self, state, roaster_id: str, sku: str,
                         output_line: Optional[str]) -> bool:
        if state.status.get(roaster_id) != "IDLE":
            return False
        if state.t in self.params["downtime_slots"].get(roaster_id, set()):
            return False
        if sku not in self.params["R_elig_skus"][roaster_id]:
            return False

        roast_t = self._roast_time[sku]
        setup_needed = self._action_requires_setup(state, roaster_id, sku)
        start_time = state.t + (self.params["sigma"] if setup_needed else 0)
        if start_time + roast_t > self.params["SL"]:
            return False
        if self._would_overlap_downtime(roaster_id, start_time, roast_t):
            return False

        pipe_line = self.params["R_pipe"][roaster_id]
        if not setup_needed and state.pipeline_busy[pipe_line] > 0:
            return False

        # GC feasibility
        gc_pair = (self.params["R_pipe"][roaster_id], sku)
        if gc_pair not in self.params.get("gc_capacity", {}):
            return False
        if state.gc_stock.get(gc_pair, 0) < 1:
            return False

        if output_line is not None and state.rc_stock[output_line] >= self.params["max_rc"]:
            return False

        return True

    def _remaining_mto_for_sku(self, state, sku: str) -> int:
        total = 0
        for job_id, job_sku in self.params["job_sku"].items():
            if job_sku == sku:
                total += int(state.mto_remaining.get(job_id, 0))
        return total

    def _min_mto_slack(self, state, sku: str) -> int | None:
        slacks = [
            int(self.params["job_due"][job_id]) - int(state.t)
            for job_id, job_sku in self.params["job_sku"].items()
            if job_sku == sku and int(state.mto_remaining.get(job_id, 0)) > 0
        ]
        return min(slacks) if slacks else None

    def _mto_due_priority(self, state, sku: str) -> int:
        slack = self._min_mto_slack(state, sku)
        if slack is None:
            return 0
        if slack <= 60:
            return 2
        if slack <= 120:
            return 1
        return 0

    def _affected_roaster_count(self, state, line_id: str, sku: str) -> int:
        count = 0
        for roaster_id in self.params["roasters"]:
            if self.params["R_pipe"][roaster_id] != line_id:
                continue
            if sku not in self.params["R_elig_skus"][roaster_id]:
                continue
            if state.status.get(roaster_id) in {"IDLE", "SETUP"}:
                count += 1
        return count

    def _future_consumption_count(self, state, line_id: str) -> int:
        return sum(1 for slot in self.params.get("consume_events", {}).get(line_id, []) if int(slot) >= int(state.t))

    def _psc_restock_relevant(self, state, line_id: str) -> bool:
        remaining_consumption = self._future_consumption_count(state, line_id)
        if remaining_consumption <= 0:
            return False
        rc = int(state.rc_stock.get(line_id, 0))
        safety = int(self.params.get("safety_stock", 0))
        return rc < safety or rc < remaining_consumption

    def _has_demand_justification(self, state, line_id: str, sku: str) -> bool:
        if sku == "PSC":
            return self._psc_restock_relevant(state, line_id)
        return self._remaining_mto_for_sku(state, sku) > 0

    # ------------------------------------------------------------------
    # Restock decision
    # ------------------------------------------------------------------

    def decide_restock(self, state) -> tuple:
        """Global restock decision using input-driven reorder points."""
        if state.restock_busy > 0:
            return ("WAIT",)

        engine = self._get_engine()
        best_pair: tuple[str, str] | None = None
        best_key: tuple[float, int, float, int, float] | None = None

        for raw_pair in self.params.get("feasible_gc_pairs", []):
            line_id, sku = tuple(raw_pair)
            pair = (line_id, sku)
            if not engine.can_start_restock(state, line_id, sku):
                continue

            current_gc = int(state.gc_stock.get(pair, 0))
            rop_impl = int(self.reorder_point_impl.get(pair, 0))
            if current_gc > rop_impl:
                continue
            if not self._has_demand_justification(state, line_id, sku):
                continue

            urgency_gap = float(rop_impl - current_gc)
            if urgency_gap < 0:
                continue

            due_priority = self._mto_due_priority(state, sku) if sku != "PSC" else 0
            cap = max(1, int(self.params["gc_capacity"][pair]))
            stock_ratio = current_gc / cap
            affected = self._affected_roaster_count(state, line_id, sku)
            min_slack = self._min_mto_slack(state, sku)
            slack_rank = -float(min_slack) if min_slack is not None else float("-inf")
            rank_key = (
                urgency_gap,
                due_priority,
                -stock_ratio,
                affected,
                slack_rank,
            )
            if best_key is None or rank_key > best_key:
                best_key = rank_key
                best_pair = pair

        if best_pair is None:
            return ("WAIT",)
        return ("START_RESTOCK", best_pair[0], best_pair[1])

    # ------------------------------------------------------------------
    # Roaster decision
    # ------------------------------------------------------------------

    def decide(self, state, roaster_id: str) -> tuple:
        """Return one action tuple for the requested roaster."""
        if state.status.get(roaster_id) != "IDLE":
            return ("WAIT",)
        if state.t in self.params["downtime_slots"].get(roaster_id, set()):
            return ("WAIT",)

        eligible_mto = []
        eligible_skus = set(self.params["R_elig_skus"][roaster_id])
        for jid in self.params["jobs"]:
            remaining = int(state.mto_remaining.get(jid, 0))
            if remaining <= 0:
                continue
            sku = self.params["job_sku"][jid]
            if sku not in eligible_skus:
                continue
            roast_t = self._roast_time[sku]
            time_left = self.params["job_due"][jid] - state.t
            setup = self.params["sigma"] if self._action_requires_setup(state, roaster_id, sku) else 0
            slots_needed = remaining * roast_t + setup
            eligible_mto.append((jid, sku, remaining, slots_needed, time_left))

        if eligible_mto:
            total_mto_time = sum(item[3] for item in eligible_mto)
            time_left = eligible_mto[0][4]
            urgency = total_mto_time / max(time_left, 1)

            if urgency >= self.MTO_URGENCY_THRESHOLD:
                eligible_mto.sort(
                    key=lambda item: (
                        -item[2],
                        0 if item[1] == "BUSTA" else 1,
                        item[0],
                    )
                )
                for _, chosen_sku, _, _, _ in eligible_mto:
                    if self._can_take_action(state, roaster_id, chosen_sku, None):
                        return (chosen_sku,)
                return ("WAIT",)

        if roaster_id == "R3" and self.params["allow_r3_flex"]:
            l1_stock = state.rc_stock["L1"]
            l2_stock = state.rc_stock["L2"]
            preferred = "L1" if l1_stock <= l2_stock else "L2"
            alternate = "L2" if preferred == "L1" else "L1"
            if self._can_take_action(state, roaster_id, "PSC", preferred):
                return ("PSC", preferred)
            if self._can_take_action(state, roaster_id, "PSC", alternate):
                return ("PSC", alternate)
            return ("WAIT",)

        output_line = self.params["R_out"][roaster_id][0]
        if not self._can_take_action(state, roaster_id, "PSC", output_line):
            return ("WAIT",)
        return ("PSC", output_line)
