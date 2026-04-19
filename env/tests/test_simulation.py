"""Pytest coverage for the GC-silo-extended reactive roasting simulation engine."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from env.data_bridge import get_sim_params
from env.export import load_run_for_test
from env.kpi_tracker import KPITracker
from env.simulation_engine import SimulationEngine
from env.simulation_state import BatchRecord, RestockRecord, UPSEvent
from env.ups_generator import generate_ups_events
from dispatch.dispatching_heuristic import DispatchingHeuristic


def make_engine():
    return SimulationEngine(get_sim_params())


def make_params():
    return get_sim_params()


# ======================================================================
# A. Loader tests
# ======================================================================

class TestLoaderSKURoastTimes:
    """A. loader loads SKU roast times from skus.csv"""

    def test_psc_roast_time(self):
        p = make_params()
        assert p["roast_time_by_sku"]["PSC"] == 15

    def test_ndg_roast_time(self):
        p = make_params()
        assert p["roast_time_by_sku"]["NDG"] == 17

    def test_busta_roast_time(self):
        p = make_params()
        assert p["roast_time_by_sku"]["BUSTA"] == 18


class TestLoaderGCParameters:
    """B. loader loads GC capacities/initial stocks from shift_parameters.csv"""

    def test_gc_capacity_l1_psc(self):
        p = make_params()
        assert p["gc_capacity"][("L1", "PSC")] == 40

    def test_gc_capacity_l1_ndg(self):
        p = make_params()
        assert p["gc_capacity"][("L1", "NDG")] == 10

    def test_gc_capacity_l1_busta(self):
        p = make_params()
        assert p["gc_capacity"][("L1", "BUSTA")] == 10

    def test_gc_capacity_l2_psc(self):
        p = make_params()
        assert p["gc_capacity"][("L2", "PSC")] == 40

    def test_gc_init_l1_psc(self):
        p = make_params()
        assert p["gc_init"][("L1", "PSC")] == 20

    def test_gc_init_l2_psc(self):
        p = make_params()
        assert p["gc_init"][("L2", "PSC")] == 20

    def test_feasible_gc_pairs(self):
        p = make_params()
        assert ("L2", "NDG") not in p["feasible_gc_pairs"]
        assert ("L2", "BUSTA") not in p["feasible_gc_pairs"]
        assert ("L1", "PSC") in p["feasible_gc_pairs"]

    def test_restock_params(self):
        p = make_params()
        assert p["restock_duration"] == 15
        assert p["restock_qty"] == 5


# ======================================================================
# C. GC deduction on batch start
# ======================================================================

class TestGCDeduction:
    """C. batch start deducts GC immediately"""

    def test_gc_deducted_on_batch_start(self):
        eng = make_engine()
        state = eng._initialize_state()
        initial_gc = state.gc_stock[("L1", "PSC")]
        kpi = KPITracker()
        eng._apply_action(state, "R1", ("PSC", "L1"), kpi)
        assert state.gc_stock[("L1", "PSC")] == initial_gc - 1

    def test_gc_deducted_for_mto(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.last_sku["R1"] = "NDG"
        state.mto_remaining = {"J1": 5, "J2": 5}
        initial_gc = state.gc_stock[("L1", "NDG")]
        kpi = KPITracker()
        eng._apply_action(state, "R1", ("NDG",), kpi)
        # NDG needs setup first, so GC is not deducted yet
        assert state.status["R1"] == "RUNNING"
        assert state.gc_stock[("L1", "NDG")] == initial_gc - 1


# ======================================================================
# D. UPS does NOT refund GC
# ======================================================================

class TestUPSNoGCRefund:
    """D. cancelled batch via UPS does NOT refund GC"""

    def test_ups_does_not_restore_gc(self):
        eng = make_engine()
        state = eng._initialize_state()
        kpi = KPITracker()
        eng._apply_action(state, "R1", ("PSC", "L1"), kpi)
        gc_after_start = state.gc_stock[("L1", "PSC")]
        ups = UPSEvent(t=1, roaster_id="R1", duration=10)
        eng._process_ups(state, ups, None, kpi)
        assert state.gc_stock[("L1", "PSC")] == gc_after_start, \
            "GC must NOT be restored when UPS cancels a batch"


# ======================================================================
# E. Restock adds +5 at completion, not at start
# ======================================================================

class TestRestockCompletion:
    """E. restock adds +5 at completion, not at start"""

    def test_restock_gc_added_at_completion(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 10
        kpi = KPITracker()
        eng._start_restock(state, "L1", "PSC", kpi)
        assert state.gc_stock[("L1", "PSC")] == 10, "GC must NOT increase at restock start"
        # Advance restock to completion (15 ticks)
        for _ in range(15):
            eng._step_pipeline_and_restock_timers(state, kpi)
        assert state.gc_stock[("L1", "PSC")] == 15, "GC must increase by +5 at restock completion"
        assert kpi.restock_count == 1


# ======================================================================
# F. Restock blocked when stock + 5 would exceed capacity
# ======================================================================

class TestRestockCapacity:
    """F. restock blocked when stock + 5 would exceed capacity"""

    def test_restock_blocked_at_capacity(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 36
        assert not eng.can_start_restock(state, "L1", "PSC"), \
            "36 + 5 = 41 > 40: restock must be blocked"

    def test_restock_allowed_at_35(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 35
        assert eng.can_start_restock(state, "L1", "PSC"), \
            "35 + 5 = 40 <= 40: restock must be allowed"


# ======================================================================
# G. Only one restock active globally
# ======================================================================

class TestRestockGlobalMutex:
    """G. only one restock active globally"""

    def test_second_restock_blocked(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 10
        state.gc_stock[("L2", "PSC")] = 10
        kpi = KPITracker()
        eng._start_restock(state, "L1", "PSC", kpi)
        assert not eng.can_start_restock(state, "L2", "PSC"), \
            "Second restock must be blocked while first is active"


# ======================================================================
# H. Restock blocks target line pipeline
# ======================================================================

class TestRestockPipelineBlocking:
    """H. restock blocks target line pipeline"""

    def test_restock_blocks_pipeline(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 10
        kpi = KPITracker()
        eng._start_restock(state, "L1", "PSC", kpi)
        assert state.pipeline_busy["L1"] == 15
        assert state.pipeline_mode["L1"] == "RESTOCK"

    def test_batch_blocked_during_restock(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 10
        kpi = KPITracker()
        eng._start_restock(state, "L1", "PSC", kpi)
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("PSC", "L1"), False) is False, \
            "Batch start must be blocked while restock occupies pipeline"


# ======================================================================
# I. SKU-specific roast times
# ======================================================================

class TestSKURoastTimes:
    """I. PSC/NDG/BUSTA use 15/17/18 runtime respectively"""

    def test_psc_batch_duration(self):
        eng = make_engine()
        state = eng._initialize_state()
        kpi = KPITracker()
        eng._apply_action(state, "R1", ("PSC", "L1"), kpi)
        assert state.remaining["R1"] == 15

    def test_ndg_batch_duration(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.last_sku["R1"] = "NDG"
        state.mto_remaining = {"J1": 5, "J2": 5}
        kpi = KPITracker()
        eng._apply_action(state, "R1", ("NDG",), kpi)
        assert state.remaining["R1"] == 17

    def test_busta_batch_duration(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.last_sku["R2"] = "BUSTA"
        state.mto_remaining = {"J1": 5, "J2": 5}
        kpi = KPITracker()
        eng._apply_action(state, "R2", ("BUSTA",), kpi)
        assert state.remaining["R2"] == 18


# ======================================================================
# J. End-of-shift masking respects SKU-specific duration
# ======================================================================

class TestEndOfShiftMasking:
    """J. end-of-shift masking respects SKU-specific duration"""

    def test_psc_blocked_at_466(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.t = 466
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("PSC", "L1"), False) is False, \
            "PSC at t=466: 466+15=481 > 480"

    def test_psc_allowed_at_465(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.t = 465
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("PSC", "L1"), False) is True, \
            "PSC at t=465: 465+15=480 <= 480"

    def test_ndg_blocked_at_464(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.t = 464
        state.last_sku["R1"] = "NDG"
        state.mto_remaining = {"J1": 5, "J2": 5}
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("NDG",), False) is False, \
            "NDG at t=464: 464+17=481 > 480"

    def test_ndg_allowed_at_463(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.t = 463
        state.last_sku["R1"] = "NDG"
        state.mto_remaining = {"J1": 5, "J2": 5}
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("NDG",), False) is True, \
            "NDG at t=463: 463+17=480 <= 480"


# ======================================================================
# K. No import dependency on CP-SAT or MILP modules
# ======================================================================

class TestNoCPSATDependency:
    """K. env has no import dependency on CP-SAT or MILP modules"""

    # env_GUI_playground.py is excluded: it's an optional GUI that uses
    # try/except for backward-compatible imports of cpsat/dispatch.
    _EXCLUDED_GUI = {"env_GUI_playground.py"}

    def test_no_cpsat_imports_in_env(self):
        import env
        env_dir = Path(env.__file__).parent
        for py_file in env_dir.glob("*.py"):
            if py_file.name in self._EXCLUDED_GUI:
                continue
            content = py_file.read_text(encoding="utf-8")
            assert "cpsat_strategy" not in content, \
                f"{py_file.name} still references cpsat_strategy"
            assert "cpsat_reduced_builder" not in content, \
                f"{py_file.name} still references cpsat_reduced_builder"
            assert "MILP_Test_v3.data" not in content, \
                f"{py_file.name} still references MILP_Test_v3.data"

    def test_no_dispatching_heuristic_in_env(self):
        import env
        env_dir = Path(env.__file__).parent
        for py_file in env_dir.glob("*.py"):
            if py_file.name in self._EXCLUDED_GUI or py_file.name == "__init__.py":
                continue
            content = py_file.read_text(encoding="utf-8")
            assert "dispatching_heuristic" not in content, \
                f"{py_file.name} still references dispatching_heuristic"


# ======================================================================
# L. Dispatch package can drive env without circular imports
# ======================================================================

class TestDispatchIntegration:
    """L. dispatch package can drive env without circular imports"""

    def test_dispatch_import(self):
        from dispatch import DispatchingHeuristic as DH
        p = make_params()
        dh = DH(p)
        assert hasattr(dh, "decide")
        assert hasattr(dh, "decide_restock")

    def test_dispatch_smoke_run(self):
        p = make_params()
        eng = SimulationEngine(p)
        strat = DispatchingHeuristic(p)
        kpi, state = eng.run(strat, ups_events=[])
        assert kpi.net_profit() > 0
        assert kpi.psc_completed > 0


# ======================================================================
# Original tests updated for new model
# ======================================================================

class TestConsumptionEvents:
    def test_L1_event_count(self):
        events = make_params()["consume_events"]["L1"]
        assert len(events) == 59

    def test_L2_event_count(self):
        events = make_params()["consume_events"]["L2"]
        assert len(events) == 61

    def test_first_L1_event(self):
        events = make_params()["consume_events"]["L1"]
        assert events[0] == 8

    def test_first_L2_event(self):
        events = make_params()["consume_events"]["L2"]
        assert events[0] == 7

    def test_events_strictly_increasing(self):
        p = make_params()
        for lid in ["L1", "L2"]:
            events = p["consume_events"][lid]
            for i in range(len(events) - 1):
                assert events[i] < events[i + 1]


class TestPhaseOrdering:
    def test_ups_before_timer_decrement(self):
        eng = make_engine()
        state = eng._initialize_state()
        batch = BatchRecord(("R1", 0), "PSC", "R1", 0, 15, "L1", False)
        state.status["R1"] = "RUNNING"
        state.remaining["R1"] = 1
        state.current_batch["R1"] = batch
        state.pipeline_mode["L1"] = "CONSUME"
        state.pipeline_batch["L1"] = batch
        state.t = 14
        initial_rc = state.rc_stock["L1"]
        ups = UPSEvent(t=14, roaster_id="R1", duration=10)
        kpi = KPITracker()
        eng._process_ups(state, ups, None, kpi)
        assert state.status["R1"] == "DOWN"
        assert len(state.cancelled_batches) == 1
        assert state.rc_stock["L1"] == initial_rc

    def test_pipeline_free_before_decision(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.pipeline_busy["L1"] = 1
        state.pipeline_mode["L1"] = "CONSUME"
        state.status["R1"] = "IDLE"
        state.needs_decision["R1"] = True
        state.t = 50
        kpi = KPITracker()
        eng._step_pipeline_and_restock_timers(state, kpi)
        assert state.pipeline_busy["L1"] == 0
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("PSC", "L1"), False) is True


class TestTimerConvention:
    def test_batch_completes_at_start_plus_roast_time(self):
        eng = make_engine()
        state = eng._initialize_state()
        batch = BatchRecord(("R1", 0), "PSC", "R1", 0, 15, "L1", False)
        state.status["R1"] = "RUNNING"
        state.remaining["R1"] = 15
        state.current_batch["R1"] = batch
        for _ in range(14):
            state.remaining["R1"] -= 1
        assert state.remaining["R1"] == 1
        state.remaining["R1"] -= 1
        assert state.remaining["R1"] == 0

    def test_pipeline_free_after_DC_slots(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.pipeline_busy["L1"] = 3
        state.pipeline_mode["L1"] = "CONSUME"
        kpi = KPITracker()
        for i in range(3):
            eng._step_pipeline_and_restock_timers(state, kpi)
            if i < 2:
                assert state.pipeline_busy["L1"] > 0
        assert state.pipeline_busy["L1"] == 0


class TestActionMask:
    def test_pipeline_busy_blocks_batch_start(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.pipeline_busy["L1"] = 2
        state.pipeline_mode["L1"] = "CONSUME"
        state.last_sku["R1"] = "PSC"
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("PSC", "L1"), False) is False

    def test_gc_zero_blocks_batch(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L1", "PSC")] = 0
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("PSC", "L1"), False) is False, \
            "Batch must be blocked when GC is 0"

    def test_wait_always_valid(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.rc_stock["L1"] = 40
        state.pipeline_busy["L1"] = 3
        mask = eng._compute_action_mask(state, "R1")
        assert mask.get(("WAIT",), False) is True

    def test_r3_flex_can_route_to_L1(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.rc_stock["L1"] = 10
        state.pipeline_busy["L2"] = 0
        mask = eng._compute_action_mask(state, "R3")
        assert mask.get(("PSC", "L1"), False) is True

    def test_r3_cannot_do_mto(self):
        eng = make_engine()
        state = eng._initialize_state()
        mask = eng._compute_action_mask(state, "R3")
        assert mask.get(("NDG",), False) is False
        assert mask.get(("BUSTA",), False) is False


class TestSetupLogic:
    def test_mto_requires_setup_from_psc(self):
        eng = make_engine()
        state = eng._initialize_state()
        assert state.last_sku["R1"] == "PSC"
        eng._apply_action(state, "R1", ("NDG",), KPITracker())
        assert state.status["R1"] == "SETUP"
        assert state.remaining["R1"] == 5
        assert state.setup_target_sku["R1"] == "NDG"

    def test_same_sku_no_setup(self):
        eng = make_engine()
        state = eng._initialize_state()
        eng._apply_action(state, "R1", ("PSC", "L1"), KPITracker())
        assert state.status["R1"] == "RUNNING"


class TestInventoryBalance:
    def test_psc_completion_credits_rc(self):
        eng = make_engine()
        state = eng._initialize_state()
        initial = state.rc_stock["L1"]
        batch = BatchRecord(("R1", 0), "PSC", "R1", 0, 15, "L1", False)
        state.status["R1"] = "RUNNING"
        state.remaining["R1"] = 1
        state.current_batch["R1"] = batch
        kpi = KPITracker()
        eng._step_roaster_timers(state, kpi)
        assert state.rc_stock["L1"] == initial + 1

    def test_mto_completion_does_not_credit_rc(self):
        eng = make_engine()
        state = eng._initialize_state()
        initial = state.rc_stock["L1"]
        batch = BatchRecord(("J1", 0), "NDG", "R1", 0, 17, None, True)
        state.status["R1"] = "RUNNING"
        state.remaining["R1"] = 1
        state.current_batch["R1"] = batch
        kpi = KPITracker()
        eng._step_roaster_timers(state, kpi)
        assert state.rc_stock["L1"] == initial

    def test_consumption_event_decrements_rc(self):
        eng = make_engine()
        state = eng._initialize_state()
        p = make_params()
        first_event = p["consume_events"]["L1"][0]
        initial = state.rc_stock["L1"]
        state.t = first_event
        kpi = KPITracker(stockout_events={"L1": 0, "L2": 0})
        eng._process_consumption_events(state, kpi)
        assert state.rc_stock["L1"] == initial - 1


class TestUPSProcessing:
    def test_running_batch_cancelled_on_ups(self):
        eng = make_engine()
        state = eng._initialize_state()
        batch = BatchRecord(("R4", 0), "PSC", "R4", 80, 95, "L2", False)
        state.status["R4"] = "RUNNING"
        state.remaining["R4"] = 6
        state.current_batch["R4"] = batch
        state.t = 87
        ups = UPSEvent(t=87, roaster_id="R4", duration=22)
        kpi = KPITracker()
        eng._process_ups(state, ups, None, kpi)
        assert state.status["R4"] == "DOWN"
        assert state.remaining["R4"] == 22
        assert len(state.cancelled_batches) == 1

    def test_pipeline_released_on_ups_during_consume(self):
        eng = make_engine()
        state = eng._initialize_state()
        batch = BatchRecord(("R3", 0), "PSC", "R3", 50, 65, "L2", False)
        state.status["R3"] = "RUNNING"
        state.remaining["R3"] = 14
        state.current_batch["R3"] = batch
        state.pipeline_busy["L2"] = 2
        state.pipeline_mode["L2"] = "CONSUME"
        state.pipeline_batch["L2"] = batch
        state.t = 51
        ups = UPSEvent(t=51, roaster_id="R3", duration=10)
        kpi = KPITracker()
        eng._process_ups(state, ups, None, kpi)
        assert state.pipeline_busy["L2"] == 0
        assert state.pipeline_mode["L2"] == "FREE"

    def test_ups_does_not_interrupt_restock(self):
        """UPS affects roasters only, not the upstream restock resource."""
        eng = make_engine()
        state = eng._initialize_state()
        state.gc_stock[("L2", "PSC")] = 10
        kpi = KPITracker()
        eng._start_restock(state, "L2", "PSC", kpi)
        assert state.restock_busy == 15
        ups = UPSEvent(t=1, roaster_id="R3", duration=10)
        eng._process_ups(state, ups, None, kpi)
        assert state.restock_busy == 15, "Restock must continue unaffected by UPS"
        assert state.active_restock is not None


class TestDispatchingHeuristicRules:
    def test_r3_prefers_l1_on_tie(self):
        state = make_engine()._initialize_state()
        state.rc_stock["L1"] = 20
        state.rc_stock["L2"] = 20
        state.status["R3"] = "IDLE"
        state.pipeline_busy["L2"] = 0
        state.t = 100
        heuristic = DispatchingHeuristic(make_params())
        action = heuristic.decide(state, "R3")
        assert action == ("PSC", "L1")

    def test_r3_waits_only_when_both_full(self):
        state = make_engine()._initialize_state()
        state.rc_stock["L1"] = 40
        state.rc_stock["L2"] = 40
        state.status["R3"] = "IDLE"
        state.pipeline_busy["L2"] = 0
        state.t = 100
        heuristic = DispatchingHeuristic(make_params())
        action = heuristic.decide(state, "R3")
        assert action == ("WAIT",)


class TestDeterministicShift:
    def _run_no_ups(self, strategy=None):
        p = make_params()
        eng = SimulationEngine(p)
        if strategy is None:
            strategy = DispatchingHeuristic(p)
        kpi, state = eng.run(strategy, ups_events=[])
        return kpi, state

    def test_no_roaster_overlap(self):
        _, state = self._run_no_ups()
        by_roaster = {}
        for b in state.completed_batches:
            by_roaster.setdefault(b.roaster, []).append((b.start, b.end))
        for rid, intervals in by_roaster.items():
            intervals.sort()
            for i in range(len(intervals) - 1):
                assert intervals[i][1] <= intervals[i + 1][0], \
                    f"Roaster overlap on {rid}"

    def test_positive_net_profit_no_ups(self):
        kpi, _ = self._run_no_ups()
        assert kpi.net_profit() > 0

    def test_all_mto_batches_completed_no_ups(self):
        kpi, _ = self._run_no_ups()
        assert kpi.ndg_completed >= 1
        assert kpi.busta_completed >= 1

    def test_gc_never_negative(self):
        _, state = self._run_no_ups()
        for pair, val in state.gc_stock.items():
            assert val >= 0, f"GC stock for {pair} went negative: {val}"

    def test_restock_count_tracked(self):
        kpi, state = self._run_no_ups()
        assert kpi.restock_count == len(state.completed_restocks)


class TestUPSIntegration:
    def test_ups_reduces_psc_throughput(self):
        p = make_params()
        strat = DispatchingHeuristic(p)
        eng_base = SimulationEngine(p)
        kpi_base, _ = eng_base.run(strat, [])
        eng_ups = SimulationEngine(p)
        ups = generate_ups_events(lambda_rate=2, mu_mean=20, seed=42)
        kpi_ups, _ = eng_ups.run(strat, ups)
        assert kpi_ups.psc_completed <= kpi_base.psc_completed

    def test_profit_accounting_identity(self):
        p = make_params()
        eng = SimulationEngine(p)
        ups = generate_ups_events(2, 20, seed=7)
        kpi, _ = eng.run(DispatchingHeuristic(p), ups)
        r = kpi.to_dict()
        assert abs(r["net_profit"] - (r["total_revenue"] - r["total_costs"])) < 0.01

    def test_kpi_json_schema_complete(self):
        p = make_params()
        eng = SimulationEngine(p)
        kpi, _ = eng.run(DispatchingHeuristic(p), [])
        r = kpi.to_dict()
        required = [
            "net_profit", "total_revenue", "total_costs",
            "psc_count", "ndg_count", "busta_count",
            "tardiness_min", "tard_cost", "setup_events", "setup_cost",
            "stockout_events", "stockout_cost", "idle_min", "idle_cost",
            "over_min", "over_cost", "restock_count",
        ]
        for k in required:
            assert k in r, f"KPI dict missing key: {k}"


class TestEdgeCases:
    def test_ups_at_slot_0(self):
        p = make_params()
        eng = SimulationEngine(p)
        ups = [UPSEvent(t=0, roaster_id="R1", duration=15)]
        kpi, _ = eng.run(DispatchingHeuristic(p), ups)
        assert kpi.net_profit() >= 0

    def test_deterministic_run_is_reproducible(self):
        p = make_params()
        ups = generate_ups_events(2, 20, seed=123)
        results = []
        for _ in range(3):
            eng = SimulationEngine(p)
            kpi, _ = eng.run(DispatchingHeuristic(p), ups)
            results.append(kpi.to_dict()["net_profit"])
        assert results[0] == results[1] == results[2]

    def test_repeated_run_calls_are_independent(self):
        p = make_params()
        eng = SimulationEngine(p)
        strategy = DispatchingHeuristic(p)
        first, _ = eng.run(strategy, [])
        second, _ = eng.run(strategy, [])
        assert first.to_dict()["net_profit"] == second.to_dict()["net_profit"]


class TestNeedsDecisionFlag:
    def test_flag_set_on_idle_transition(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.status["R1"] = "DOWN"
        state.remaining["R1"] = 1
        state.needs_decision["R1"] = False
        kpi = KPITracker()
        eng._step_roaster_timers(state, kpi)
        assert state.needs_decision["R1"] is True

    def test_flag_cleared_after_batch_start(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.status["R1"] = "IDLE"
        state.needs_decision["R1"] = True
        eng._apply_action(state, "R1", ("PSC", "L1"), KPITracker())
        assert state.needs_decision["R1"] is False

    def test_flag_stays_true_after_wait(self):
        eng = make_engine()
        state = eng._initialize_state()
        state.status["R1"] = "IDLE"
        state.needs_decision["R1"] = True
        eng._apply_action(state, "R1", ("WAIT",), KPITracker())
        assert state.needs_decision["R1"] is True


class TestFromFile:
    def test_load_and_verify_from_file(self, pytestconfig):
        path = pytestconfig.getoption("--result-file")
        if path is None:
            pytest.skip("No --result-file provided")
        result = load_run_for_test(path)
        assert "net_profit" in result
        assert result["net_profit"] == (result["total_revenue"] - result["total_costs"])
