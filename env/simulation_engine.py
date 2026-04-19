"""Core slot-by-slot simulation engine with GC silo inventory and restock logic.

Phase ordering per slot:
  1. UPS events
  2. Roaster timers (batch/setup/down completion)
  3. Pipeline + restock timers (consume release, restock completion -> GC deposit)
  4. RC consumption events + stockout KPI
  5. Idle/overflow penalty accrual
  6a. Global restock decision point
  6b. Per-roaster decision points
"""

from __future__ import annotations

from collections import defaultdict, deque
import logging
import time
from typing import Optional

from .data_bridge import get_sim_params
from .kpi_tracker import KPITracker
from .simulation_state import BatchRecord, RestockRecord, SimulationState, UPSEvent

logger = logging.getLogger("simulation_engine")


class SimulationEngine:
    """Strategy-agnostic simulation with GC silo inventory and restock support."""

    def __init__(self, params: dict):
        self.params = params
        self.roasters = list(params["roasters"])
        self.lines = list(params["lines"])
        self._roast_time = dict(params["roast_time_by_sku"])
        self._event_slots = {
            lid: set(params["consume_events"][lid]) for lid in self.lines
        }
        self._downtime_slots = {
            rid: set(params["downtime_slots"].get(rid, set()))
            for rid in self.roasters
        }
        self._job_by_sku = {
            params["job_sku"][jid]: jid for jid in params["jobs"]
        }
        self._reset_run_context()

    # ------------------------------------------------------------------
    # Run context
    # ------------------------------------------------------------------

    def _reset_run_context(self):
        self._psc_available = {
            rid: deque(bid for bid in self.params["psc_pool"] if bid[0] == rid)
            for rid in self.roasters
        }
        self._mto_available = {
            jid: deque(bid for bid in self.params["mto_batches"] if bid[0] == jid)
            for jid in self.params["jobs"]
        }
        self._mto_completed_count = {jid: 0 for jid in self.params["jobs"]}
        self._mto_latest_end = {jid: 0 for jid in self.params["jobs"]}
        self._ups_by_time: dict[int, list[UPSEvent]] = defaultdict(list)

    def _make_kpi_tracker(self) -> KPITracker:
        return KPITracker(
            tardiness_min={jid: 0.0 for jid in self.params["jobs"]},
            stockout_events={lid: 0 for lid in self.lines},
            stockout_duration={lid: 0 for lid in self.lines},
            idle_min_per_roaster={rid: 0.0 for rid in self.roasters},
            over_min_per_roaster={rid: 0.0 for rid in self.roasters},
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        strategy,
        ups_events: list,
        schedule: dict | None = None,
    ) -> tuple[KPITracker, SimulationState]:
        """Run one fresh shift simulation and return KPI/state results.

        ``schedule=...`` used to populate a queue that was never consumed.
        Keep the argument only to fail loudly until a real planner-to-env
        execution path is implemented.
        """

        if schedule is not None:
            raise NotImplementedError(
                "SimulationEngine.run(schedule=...) is disabled. "
                "Use a live strategy interface instead."
            )

        state = self._initialize_state()

        kpi = self._make_kpi_tracker()
        self._ups_by_time.clear()
        for ev in sorted(ups_events, key=lambda e: (e.t, e.roaster_id, e.duration)):
            self._ups_by_time[ev.t].append(ev)

        for slot in range(self.params["SL"]):
            state.t = slot
            for ev in self._ups_by_time.get(slot, []):
                self._process_ups(state, ev, strategy, kpi)
            self._step_roaster_timers(state, kpi)
            self._step_pipeline_and_restock_timers(state, kpi)
            self._process_consumption_events(state, kpi)
            self._track_stockout_duration(state, kpi)
            self._accrue_idle_penalties(state, kpi)
            self._process_restock_decision_point(state, strategy, kpi)
            self._process_decision_points(state, strategy, kpi)
            state.trace.append(self.get_state_snapshot(state))

        # End-of-shift: penalize unfinished MTO batches
        self._penalize_skipped_mto(state, kpi)

        return kpi, state

    def _penalize_skipped_mto(self, state: SimulationState, kpi: KPITracker) -> None:
        """Apply penalty for each MTO batch that was not completed by end of shift.

        Per the mathematical model: c_skip = $50,000 per unscheduled MTO batch.
        """
        c_skip = float(self.params.get("c_skip_mto", 50000))
        skipped = 0
        for jid in self.params.get("jobs", {}):
            remaining = state.mto_remaining.get(jid, 0)
            if remaining > 0:
                skipped += remaining
                kpi.tard_cost += remaining * c_skip
        if skipped > 0:
            kpi.mto_skipped = skipped

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_state(self) -> SimulationState:
        self._reset_run_context()
        return SimulationState(
            t=0,
            status={rid: "IDLE" for rid in self.roasters},
            remaining={rid: 0 for rid in self.roasters},
            current_batch={rid: None for rid in self.roasters},
            last_sku=dict(self.params["roaster_initial_sku"]),
            setup_target_sku={rid: None for rid in self.roasters},
            pipeline_busy={lid: 0 for lid in self.lines},
            pipeline_mode={lid: "FREE" for lid in self.lines},
            pipeline_batch={lid: None for lid in self.lines},
            rc_stock=dict(self.params["rc_init"]),
            gc_stock=dict(self.params["gc_init"]),
            restock_busy=0,
            active_restock=None,
            needs_decision={rid: True for rid in self.roasters},
            needs_restock_decision=True,
            schedule_queue={rid: deque() for rid in self.roasters},
            mto_remaining={
                jid: int(self.params["job_batches"][jid])
                for jid in self.params["jobs"]
            },
            mto_tardiness={jid: 0.0 for jid in self.params["jobs"]},
            completed_batches=[],
            cancelled_batches=[],
            completed_restocks=[],
            ups_events_fired=[],
            trace=[],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def gc_pair_of_batch(self, roaster_id: str, sku: str) -> tuple[str, str]:
        """Return the (line, sku) GC silo pair that a batch on this roaster draws from."""
        return (self.params["R_pipe"][roaster_id], sku)

    def can_start_batch(self, state: SimulationState, roaster_id: str,
                        sku: str, output_line: Optional[str]) -> bool:
        """Check all hard constraints for starting a batch (no setup needed)."""
        if state.status.get(roaster_id) != "IDLE":
            return False
        if sku not in self.params["R_elig_skus"][roaster_id]:
            return False

        pipe_line = self.params["R_pipe"][roaster_id]
        if state.pipeline_busy[pipe_line] > 0 and state.pipeline_mode[pipe_line] != "FREE":
            return False
        if state.pipeline_busy[pipe_line] > 0:
            return False

        roast_t = self._roast_time[sku]
        if state.t + roast_t > self.params["SL"]:
            return False

        gc_pair = self.gc_pair_of_batch(roaster_id, sku)
        if gc_pair not in self.params["gc_capacity"]:
            return False
        if state.gc_stock.get(gc_pair, 0) < 1:
            return False

        if sku == "PSC" and output_line is not None:
            projected = self._projected_rc_after_psc_completion(
                state, output_line, state.t + roast_t,
            )
            if projected > self.params["max_rc"]:
                return False

        if self._would_overlap_downtime(roaster_id, state.t, roast_t):
            return False

        return True

    def can_start_restock(self, state: SimulationState,
                          line_id: str, sku: str) -> bool:
        """Check all hard constraints for starting a restock."""
        pair = (line_id, sku)
        if pair not in self.params["gc_capacity"]:
            return False
        if state.restock_busy > 0:
            return False
        if state.pipeline_busy[line_id] > 0:
            return False
        rst_dur = self.params["restock_duration"]
        if state.t + rst_dur > self.params["SL"]:
            return False
        rst_qty = self.params["restock_qty"]
        if state.gc_stock.get(pair, 0) + rst_qty > self.params["gc_capacity"][pair]:
            return False
        return True

    # ------------------------------------------------------------------
    # UPS processing
    # ------------------------------------------------------------------

    def _process_ups(self, state, event, strategy, kpi):
        rid = event.roaster_id
        pipe_line = self.params["R_pipe"][rid]
        status = state.status[rid]
        state.ups_events_fired.append(event)

        if status == "RUNNING":
            batch = state.current_batch[rid]
            if batch is not None:
                state.cancelled_batches.append(batch)
                # GC NOT restored — already consumed at batch start
                if batch.is_mto:
                    jid = batch.batch_id[0]
                    self._mto_available[jid].appendleft(batch.batch_id)
                    state.mto_remaining[jid] = state.mto_remaining.get(jid, 0) + 1
                if (
                    state.pipeline_mode[pipe_line] == "CONSUME"
                    and state.pipeline_batch[pipe_line] is batch
                    and state.pipeline_busy[pipe_line] > 0
                ):
                    state.pipeline_busy[pipe_line] = 0
                    state.pipeline_mode[pipe_line] = "FREE"
                    state.pipeline_batch[pipe_line] = None
            state.current_batch[rid] = None
        elif status == "SETUP":
            state.setup_target_sku[rid] = None
        elif status == "DOWN":
            state.remaining[rid] = max(state.remaining[rid], int(event.duration))
            state.needs_decision[rid] = False
            if strategy is not None and hasattr(strategy, "on_ups"):
                t0 = time.perf_counter()
                strategy.on_ups(state, event)
                kpi.total_compute_ms += (time.perf_counter() - t0) * 1000.0
                kpi.num_resolves += 1
            return

        state.status[rid] = "DOWN"
        state.remaining[rid] = int(event.duration)
        state.current_batch[rid] = None
        state.setup_target_sku[rid] = None
        state.needs_decision[rid] = False

        if strategy is not None and hasattr(strategy, "on_ups"):
            t0 = time.perf_counter()
            strategy.on_ups(state, event)
            kpi.total_compute_ms += (time.perf_counter() - t0) * 1000.0
            kpi.num_resolves += 1

    # ------------------------------------------------------------------
    # Timer advancement
    # ------------------------------------------------------------------

    def _step_roaster_timers(self, state, kpi):
        for rid in self.roasters:
            status = state.status[rid]
            if status not in {"RUNNING", "SETUP", "DOWN"}:
                continue
            if state.remaining[rid] <= 0:
                continue

            state.remaining[rid] -= 1
            if state.remaining[rid] != 0:
                continue

            if status == "RUNNING":
                batch = state.current_batch[rid]
                if batch is not None:
                    state.completed_batches.append(batch)
                    state.last_sku[rid] = batch.sku

                    if batch.sku == "PSC":
                        kpi.psc_completed += 1
                        kpi.revenue_psc += float(self.params["rev_psc"])
                        kpi.total_revenue += float(self.params["rev_psc"])
                    elif batch.sku == "NDG":
                        kpi.ndg_completed += 1
                        kpi.revenue_ndg += float(self.params["rev_ndg"])
                        kpi.total_revenue += float(self.params["rev_ndg"])
                    elif batch.sku == "BUSTA":
                        kpi.busta_completed += 1
                        kpi.revenue_busta += float(self.params["rev_busta"])
                        kpi.total_revenue += float(self.params["rev_busta"])

                    if batch.output_line is not None and self.params["sku_credits_rc"][batch.sku]:
                        state.rc_stock[batch.output_line] += 1

                    if batch.is_mto:
                        jid = batch.batch_id[0]
                        self._mto_completed_count[jid] = (
                            self._mto_completed_count.get(jid, 0) + 1
                        )
                        # Track latest end time per job: tard_j = max(0, max_b(e_b) - D)
                        if batch.end > self._mto_latest_end.get(jid, 0):
                            self._mto_latest_end[jid] = batch.end
                        # Update tardiness based on latest-finishing batch so far
                        tardiness = max(0, self._mto_latest_end[jid] - self.params["job_due"][jid])
                        previous = float(kpi.tardiness_min.get(jid, 0.0))
                        kpi.tard_cost -= previous * float(self.params["c_tard"])
                        state.mto_tardiness[jid] = float(tardiness)
                        kpi.tardiness_min[jid] = float(tardiness)
                        kpi.tard_cost += float(tardiness) * float(self.params["c_tard"])

                state.current_batch[rid] = None
                state.status[rid] = "IDLE"
                state.needs_decision[rid] = True
            elif status == "SETUP":
                target = state.setup_target_sku[rid]
                if target is not None:
                    state.last_sku[rid] = target
                state.setup_target_sku[rid] = None
                state.status[rid] = "IDLE"
                state.needs_decision[rid] = True
            elif status == "DOWN":
                state.status[rid] = "IDLE"
                state.needs_decision[rid] = True

    def _step_pipeline_and_restock_timers(self, state, kpi):
        """Advance pipeline occupancy and handle restock completion (GC deposit)."""
        for lid in self.lines:
            if state.pipeline_busy[lid] <= 0:
                continue
            state.pipeline_busy[lid] -= 1
            if state.pipeline_busy[lid] == 0:
                if state.pipeline_mode[lid] == "RESTOCK":
                    rst = state.active_restock
                    if rst is not None and rst.line_id == lid:
                        pair = (rst.line_id, rst.sku)
                        state.gc_stock[pair] = state.gc_stock.get(pair, 0) + rst.qty
                        state.completed_restocks.append(rst)
                        kpi.restock_count += 1
                state.pipeline_mode[lid] = "FREE"
                state.pipeline_batch[lid] = None

        if state.restock_busy > 0:
            state.restock_busy -= 1
            if state.restock_busy == 0:
                state.active_restock = None
                state.needs_restock_decision = True

    # ------------------------------------------------------------------
    # RC consumption
    # ------------------------------------------------------------------

    def _process_consumption_events(self, state, kpi):
        for lid in self.lines:
            if state.t not in self._event_slots[lid]:
                continue
            state.rc_stock[lid] -= 1
            if state.rc_stock[lid] < 0:
                kpi.stockout_events[lid] += 1
                kpi.stockout_cost += float(self.params["c_stock"])

    def _track_stockout_duration(self, state, kpi):
        for lid in self.lines:
            if state.rc_stock[lid] <= 0:
                kpi.stockout_duration[lid] += 1

    # ------------------------------------------------------------------
    # Idle / overflow penalties
    # ------------------------------------------------------------------

    def _accrue_idle_penalties(self, state, kpi):
        for rid in self.roasters:
            if state.t in self._downtime_slots[rid]:
                continue

            home_line = self.params["R_line"][rid]
            status = state.status[rid]
            if status in {"IDLE", "SETUP"} and state.rc_stock[home_line] < self.params["safety_stock"]:
                kpi.idle_min_per_roaster[rid] = (
                    kpi.idle_min_per_roaster.get(rid, 0.0) + 1
                )
                kpi.idle_cost += float(self.params["c_idle"])

            if status != "IDLE":
                continue
            if rid == "R3" and self.params["allow_r3_flex"]:
                if (
                    state.rc_stock["L1"] >= self.params["max_rc"]
                    and state.rc_stock["L2"] >= self.params["max_rc"]
                ):
                    kpi.over_min_per_roaster[rid] = (
                        kpi.over_min_per_roaster.get(rid, 0.0) + 1
                    )
                    kpi.over_cost += float(self.params["c_over"])
            else:
                out_line = self.params["R_out"][rid][0]
                if state.rc_stock[out_line] >= self.params["max_rc"]:
                    kpi.over_min_per_roaster[rid] = (
                        kpi.over_min_per_roaster.get(rid, 0.0) + 1
                    )
                    kpi.over_cost += float(self.params["c_over"])

    # ------------------------------------------------------------------
    # Decision points
    # ------------------------------------------------------------------

    def _process_restock_decision_point(self, state, strategy, kpi):
        """Global restock decision point — fires when no restock is active."""
        if state.restock_busy > 0:
            return
        if strategy is None or not hasattr(strategy, "decide_restock"):
            return

        action = strategy.decide_restock(state)
        if action is None or action == ("WAIT",):
            return

        if len(action) == 3 and action[0] == "START_RESTOCK":
            _, line_id, sku = action
            if self.can_start_restock(state, line_id, sku):
                self._start_restock(state, line_id, sku, kpi)

    def _process_decision_points(self, state, strategy, kpi):
        for rid in self.roasters:
            if state.status[rid] != "IDLE" or not state.needs_decision[rid]:
                continue
            mask = self._compute_action_mask(state, rid)
            action = ("WAIT",)
            if strategy is not None and hasattr(strategy, "decide"):
                action = strategy.decide(state, rid)
            if not mask.get(action, False):
                action = ("WAIT",)
            self._apply_action(state, rid, action, kpi)

    # ------------------------------------------------------------------
    # Action mask
    # ------------------------------------------------------------------

    def _would_overlap_downtime(self, roaster_id: str, start: int, duration: int) -> bool:
        if duration <= 0:
            return False
        end_excl = min(start + duration, self.params["SL"])
        for slot in range(start, end_excl):
            if slot in self._downtime_slots[roaster_id]:
                return True
        return False

    def _action_requires_setup(self, state, roaster_id: str, sku: str) -> bool:
        return state.last_sku[roaster_id] != sku

    def _projected_rc_after_psc_completion(self, state, output_line: str,
                                           completion_time: int) -> int:
        future_consumption = sum(
            1 for tau in self.params["consume_events"][output_line]
            if state.t < tau <= completion_time
        )
        future_completions = 0
        for rid in self.roasters:
            if state.status[rid] != "RUNNING":
                continue
            batch = state.current_batch[rid]
            if batch is None or batch.sku != "PSC" or batch.output_line != output_line:
                continue
            bc = state.t + int(state.remaining[rid])
            if state.t < bc <= completion_time:
                future_completions += 1
        return int(state.rc_stock[output_line]) + future_completions - future_consumption + 1

    def _compute_action_mask(self, state, roaster_id: str) -> dict:
        mask: dict[tuple, bool] = {("WAIT",): True}
        is_idle = state.status.get(roaster_id) == "IDLE"
        pipe_line = self.params["R_pipe"][roaster_id]
        pipeline_free = state.pipeline_busy[pipe_line] == 0

        def _valid_now(sku: str, output_line: Optional[str]) -> bool:
            if not is_idle:
                return False
            if sku not in self.params["R_elig_skus"][roaster_id]:
                return False

            setup_needed = self._action_requires_setup(state, roaster_id, sku)
            if not setup_needed and not pipeline_free:
                return False

            roast_t = self._roast_time[sku]
            start_time = state.t + (self.params["sigma"] if setup_needed else 0)
            completion_time = start_time + roast_t
            if completion_time > self.params["SL"]:
                return False

            gc_pair = self.gc_pair_of_batch(roaster_id, sku)
            if gc_pair not in self.params["gc_capacity"]:
                return False
            if state.gc_stock.get(gc_pair, 0) < 1:
                return False

            if sku == "PSC" and output_line is not None:
                projected = self._projected_rc_after_psc_completion(
                    state, output_line, completion_time,
                )
                if projected > self.params["max_rc"]:
                    return False

            if self._would_overlap_downtime(roaster_id, start_time, roast_t):
                return False
            return True

        if "PSC" in self.params["R_elig_skus"][roaster_id] and self._psc_available[roaster_id]:
            if roaster_id == "R3" and self.params["allow_r3_flex"]:
                for ol in ("L1", "L2"):
                    mask[("PSC", ol)] = _valid_now("PSC", ol)
            else:
                ol = self.params["R_out"][roaster_id][0]
                mask[("PSC", ol)] = _valid_now("PSC", ol)

        for jid in self.params["jobs"]:
            if state.mto_remaining.get(jid, 0) <= 0:
                continue
            if not self._mto_available[jid]:
                continue
            sku = self.params["job_sku"][jid]
            mask[(sku,)] = _valid_now(sku, None)

        return mask

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    def _next_psc_batch_id(self, roaster_id: str):
        if not self._psc_available[roaster_id]:
            return None
        return self._psc_available[roaster_id].popleft()

    def _next_mto_batch_id(self, sku: str):
        jid = self._job_by_sku[sku]
        if not self._mto_available[jid]:
            return None
        return self._mto_available[jid].popleft()

    def _apply_action(self, state, roaster_id, action, kpi):
        if not self._compute_action_mask(state, roaster_id).get(action, False):
            action = ("WAIT",)

        if action == ("WAIT",):
            state.status[roaster_id] = "IDLE"
            state.needs_decision[roaster_id] = True
            return

        sku = action[0]
        output_line = action[1] if sku == "PSC" else None

        if self._action_requires_setup(state, roaster_id, sku):
            state.status[roaster_id] = "SETUP"
            state.remaining[roaster_id] = int(self.params["sigma"])
            state.setup_target_sku[roaster_id] = sku
            state.current_batch[roaster_id] = None
            state.needs_decision[roaster_id] = False
            kpi.setup_events += 1
            kpi.setup_cost += float(self.params["c_setup"])
            return

        if sku == "PSC":
            batch_id = self._next_psc_batch_id(roaster_id)
            is_mto = False
        else:
            batch_id = self._next_mto_batch_id(sku)
            is_mto = True

        if batch_id is None:
            state.status[roaster_id] = "IDLE"
            state.needs_decision[roaster_id] = True
            return

        if is_mto:
            jid = self._job_by_sku[sku]
            state.mto_remaining[jid] = max(0, state.mto_remaining.get(jid, 0) - 1)

        roast_t = self._roast_time[sku]
        batch = BatchRecord(
            batch_id=batch_id,
            sku=sku,
            roaster=roaster_id,
            start=state.t,
            end=state.t + roast_t,
            output_line=output_line,
            is_mto=is_mto,
        )

        # GC deduction at batch start
        gc_pair = self.gc_pair_of_batch(roaster_id, sku)
        state.gc_stock[gc_pair] -= 1

        pipe_line = self.params["R_pipe"][roaster_id]
        state.status[roaster_id] = "RUNNING"
        state.remaining[roaster_id] = roast_t
        state.current_batch[roaster_id] = batch
        state.setup_target_sku[roaster_id] = None
        state.pipeline_busy[pipe_line] = int(self.params["DC"])
        state.pipeline_mode[pipe_line] = "CONSUME"
        state.pipeline_batch[pipe_line] = batch
        state.needs_decision[roaster_id] = False

    def _start_restock(self, state, line_id: str, sku: str, kpi):
        """Execute a validated restock start."""
        rst_dur = self.params["restock_duration"]
        rst_qty = self.params["restock_qty"]
        rst = RestockRecord(
            line_id=line_id, sku=sku,
            start=state.t, end=state.t + rst_dur,
            qty=rst_qty,
        )
        state.restock_busy = rst_dur
        state.active_restock = rst
        state.pipeline_busy[line_id] = rst_dur
        state.pipeline_mode[line_id] = "RESTOCK"
        state.pipeline_batch[line_id] = (line_id, sku)
        state.needs_restock_decision = False

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_to_dict(batch):
        if batch is None:
            return None
        return {
            "batch_id": str(batch.batch_id),
            "sku": batch.sku,
            "roaster": batch.roaster,
            "start": int(batch.start),
            "end": int(batch.end),
            "output_line": batch.output_line,
            "is_mto": bool(batch.is_mto),
        }

    def get_state_snapshot(self, state) -> dict:
        return {
            "t": int(state.t),
            "status": dict(state.status),
            "remaining": dict(state.remaining),
            "current_batch": {
                rid: self._batch_to_dict(b)
                for rid, b in state.current_batch.items()
            },
            "last_sku": dict(state.last_sku),
            "setup_target_sku": dict(state.setup_target_sku),
            "pipeline_busy": dict(state.pipeline_busy),
            "pipeline_mode": dict(state.pipeline_mode),
            "rc_stock": dict(state.rc_stock),
            "gc_stock": {f"{k[0]}_{k[1]}": v for k, v in state.gc_stock.items()},
            "restock_busy": state.restock_busy,
            "needs_decision": dict(state.needs_decision),
            "schedule_queue_len": {
                rid: len(q) for rid, q in state.schedule_queue.items()
            },
            "completed_count": len(state.completed_batches),
            "cancelled_count": len(state.cancelled_batches),
            "restock_count": len(state.completed_restocks),
        }
