"""
Core discrete-time simulation engine for the coffee roasting production line.
Runs a 480-minute shift with 1-minute time steps.
"""

from simulation.roaster import (
    Roaster, RoasterState, ROAST_TIME, SETUP_TIME, CONSUME_TIME,
    SKU_PSC, SKU_NDG, SKU_BUSTA, MAX_SHIFT_TIME
)
from simulation.inventory import RCSilo
from simulation.disruption import DisruptionManager


class SimulationEngine:
    def __init__(self, config=None):
        config = config or {}

        # --- Initialize roasters ---
        self.roasters = {
            rid: Roaster(rid) for rid in ["R1", "R2", "R3", "R4", "R5"]
        }

        # --- Initialize RC silos ---
        self.silos = {
            "L1": RCSilo(
                "L1",
                initial_stock=config.get("rc_init_L1", 20),
                max_buffer=config.get("rc_max_L1", 40),
                consumption_rate=config.get("consumption_rate_L1", 5.1),
            ),
            "L2": RCSilo(
                "L2",
                initial_stock=config.get("rc_init_L2", 20),
                max_buffer=config.get("rc_max_L2", 40),
                consumption_rate=config.get("consumption_rate_L2", 4.8),
            ),
        }

        # --- Pipelines (one per line) ---
        self.pipeline_busy = {"L1": 0, "L2": 0}
        self.pipeline_batch_roaster = {"L1": None, "L2": None}

        # --- Disruption manager ---
        self.disruptions = DisruptionManager(seed=config.get("seed", None))
        lam = config.get("disruption_lambda", 0)
        if lam > 0:
            self.disruptions.generate_random_events(
                lam=lam,
                min_duration=config.get("disruption_min", 10),
                max_duration=config.get("disruption_max", 30),
            )

        # --- MTO tracking ---
        self.mto_jobs = {
            "NDG": {"remaining": config.get("mto_ndg", 3), "completed": 0, "completed_time": None},
            "Busta": {"remaining": config.get("mto_busta", 1), "completed": 0, "completed_time": None},
        }
        self.mto_due_date = 240  # Slot 240

        # --- Simulation state ---
        self.current_time = 0
        self.running = False
        self.finished = False
        self.auto_mode = False
        self.speed = 1  # 1x, 5x, 10x, 50x

        # --- Event log ---
        self.event_log = []

        # --- Chart data ---
        self.rc_history = {"L1": [], "L2": []}
        self.utilization_history = []
        self.batch_history = []

        # --- AI Decision log ---
        self.decision_log = []

    def _record_state(self):
        """Record current state for charts."""
        t = self.current_time
        self.rc_history["L1"].append({"t": t, "stock": self.silos["L1"].stock})
        self.rc_history["L2"].append({"t": t, "stock": self.silos["L2"].stock})

    def step(self, n=1):
        """Advance simulation by n time steps."""
        results = []
        for _ in range(n):
            if self.current_time >= MAX_SHIFT_TIME or self.finished:
                self.finished = True
                break
            result = self._tick()
            results.append(result)
            # Record state AFTER all phases (including consumption) so chart matches silo
            self._record_state()
            self.current_time += 1
        return results

    def _tick(self):
        """Execute one simulation tick. 5 phases per tick."""
        t = self.current_time
        tick_events = []

        # --- Phase 1: Process disruption events ---
        disruption_events = self.disruptions.get_events_at(t)
        # Also process any pending manual events
        manual = self.disruptions.pop_manual_events()
        disruption_events.extend(manual)

        for event in disruption_events:
            rid = event["roaster"]
            roaster = self.roasters[rid]
            cancelled = roaster.trigger_breakdown(event["duration"], t)
            self.disruptions.log_event(event, t)

            # If a batch was cancelled, handle pipeline release
            if cancelled:
                pipeline = roaster.pipeline
                if self.pipeline_batch_roaster[pipeline] == rid:
                    self.pipeline_busy[pipeline] = 0
                    self.pipeline_batch_roaster[pipeline] = None

                # If MTO was cancelled, return to remaining
                if cancelled == SKU_NDG:
                    self.mto_jobs["NDG"]["remaining"] += 1
                elif cancelled == SKU_BUSTA:
                    self.mto_jobs["Busta"]["remaining"] += 1

            tick_events.append({
                "type": "disruption",
                "roaster": rid,
                "duration": event["duration"],
                "source": event["type"],
                "time": t,
            })

        # --- Phase 2: Advance roaster timers ---
        for rid, roaster in self.roasters.items():
            events = roaster.tick(t)
            for ev in events:
                if ev["type"] == "batch_complete":
                    sku = ev["sku"]
                    if sku == SKU_PSC:
                        target_line = ev.get("target_line", roaster.line)
                        self.silos[target_line].add_batch(t)
                    elif sku == SKU_NDG:
                        self.mto_jobs["NDG"]["completed"] += 1
                        self.mto_jobs["NDG"]["remaining"] -= 1
                        self.mto_jobs["NDG"]["completed_time"] = t
                    elif sku == SKU_BUSTA:
                        self.mto_jobs["Busta"]["completed"] += 1
                        self.mto_jobs["Busta"]["remaining"] -= 1
                        self.mto_jobs["Busta"]["completed_time"] = t

                    self.batch_history.append({
                        "roaster": rid,
                        "sku": sku,
                        "completed_at": t,
                    })
                    tick_events.append(ev)

                elif ev["type"] == "setup_complete":
                    tick_events.append(ev)

                elif ev["type"] == "repair_complete":
                    tick_events.append(ev)

        # --- Phase 3: Advance pipeline timers ---
        for line in ["L1", "L2"]:
            if self.pipeline_busy[line] > 0:
                self.pipeline_busy[line] -= 1
                if self.pipeline_busy[line] <= 0:
                    self.pipeline_batch_roaster[line] = None

        # --- Phase 4: Process consumption events ---
        for line_id, silo in self.silos.items():
            consumed = silo.process_consumption(t)
            if consumed and silo.stock < 0:
                tick_events.append({
                    "type": "stockout",
                    "line": line_id,
                    "stock": silo.stock,
                    "time": t,
                })

        # --- Phase 5: Process decision points (auto mode) ---
        if self.auto_mode:
            from scheduler.dispatch_rule import auto_dispatch
            decisions = auto_dispatch(self, t)
            if decisions:
                self.decision_log.extend(decisions)

        # Process batch queues for roasters that just became idle
        for rid, roaster in self.roasters.items():
            if roaster.state == RoasterState.IDLE and roaster.pending_sku:
                # Setup completed — start roasting
                pipeline = roaster.pipeline
                if self.pipeline_busy[pipeline] <= 0:
                    target_line = roaster.line
                    if roaster.batch_queue:
                        _, tl = roaster.batch_queue[0]
                        if tl:
                            target_line = tl
                    roaster.start_roasting_after_setup(t, target_line)
                    self.pipeline_busy[pipeline] = CONSUME_TIME
                    self.pipeline_batch_roaster[pipeline] = rid

        self.event_log.extend(tick_events)
        return tick_events

    def schedule_batch(self, roaster_id, sku, target_line=None):
        """User manually schedules a batch on a roaster."""
        roaster = self.roasters.get(roaster_id)
        if not roaster:
            return {"error": f"Roaster {roaster_id} not found"}

        if roaster.state != RoasterState.IDLE:
            return {"error": f"Roaster {roaster_id} is {roaster.state.value}"}

        if not roaster.can_produce(sku):
            return {"error": f"Roaster {roaster_id} cannot produce {sku}"}

        # Check pipeline availability
        pipeline = roaster.pipeline
        if roaster.needs_setup(sku):
            # Setup doesn't need pipeline — we can start setup even if pipeline busy
            pass
        else:
            if self.pipeline_busy[pipeline] > 0:
                return {"error": f"Pipeline {pipeline} is busy ({self.pipeline_busy[pipeline]} min remaining)"}

        # Check shift time constraint
        setup = SETUP_TIME if roaster.needs_setup(sku) else 0
        if self.current_time + setup + ROAST_TIME > MAX_SHIFT_TIME:
            return {"error": "Not enough time remaining in shift"}

        # Check MTO remaining
        if sku in [SKU_NDG, SKU_BUSTA]:
            job_key = sku
            if self.mto_jobs[job_key]["remaining"] <= 0:
                return {"error": f"No remaining {sku} MTO batches to schedule"}

        # Check overflow
        if sku == SKU_PSC:
            out_line = target_line or roaster.line
            if roaster.id == "R3" and target_line is None:
                out_line = roaster.line
            if self.silos[out_line].would_overflow():
                return {"error": f"RC silo {out_line} would overflow"}

        # Start the batch
        if target_line is None and roaster.id == "R3":
            # Default R3 routing: to whichever line has lower stock
            if self.silos["L1"].stock <= self.silos["L2"].stock:
                target_line = "L1"
            else:
                target_line = "L2"

        started = roaster.start_batch(sku, self.current_time, target_line)
        if started:
            if roaster.state == RoasterState.ROASTING:
                # Occupy pipeline
                self.pipeline_busy[pipeline] = CONSUME_TIME
                self.pipeline_batch_roaster[pipeline] = roaster_id
            elif roaster.state == RoasterState.SETUP:
                # Queue the batch info for after setup
                roaster.batch_queue = [(sku, target_line)]

            return {"success": True, "message": f"Scheduled {sku} on {roaster_id}"}
        else:
            return {"error": "Failed to start batch"}

    def trigger_disruption(self, roaster_id, duration=20):
        """User manually triggers a disruption."""
        self.disruptions.add_manual_disruption(roaster_id, duration, self.current_time)
        # Process immediately
        roaster = self.roasters[roaster_id]
        cancelled = roaster.trigger_breakdown(duration, self.current_time)

        if cancelled:
            pipeline = roaster.pipeline
            if self.pipeline_batch_roaster[pipeline] == roaster_id:
                self.pipeline_busy[pipeline] = 0
                self.pipeline_batch_roaster[pipeline] = None
            if cancelled == SKU_NDG:
                self.mto_jobs["NDG"]["remaining"] += 1
            elif cancelled == SKU_BUSTA:
                self.mto_jobs["Busta"]["remaining"] += 1

        self.disruptions.log_event({
            "time": self.current_time,
            "roaster": roaster_id,
            "duration": duration,
            "type": "manual",
        }, self.current_time)

        return {"success": True, "message": f"Disruption triggered on {roaster_id} for {duration} min"}

    def generate_report(self):
        """Generate end-of-shift report."""
        total_psc = sum(s.total_produced for s in self.silos.values())

        # MTO tardiness
        mto_tardiness = 0
        for job_key, job in self.mto_jobs.items():
            if job["completed_time"] and job["completed_time"] > self.mto_due_date:
                mto_tardiness += job["completed_time"] - self.mto_due_date

        # Stockout
        total_stockout_events = sum(s.stockout_count for s in self.silos.values())
        total_stockout_duration = sum(s.stockout_duration for s in self.silos.values())

        # Roaster utilization
        utilization = {}
        for rid, r in self.roasters.items():
            utilization[rid] = {
                "utilization": round(r.get_utilization(self.current_time), 1),
                "batches": r.batches_completed,
                "roast_time": r.total_roasting_time,
                "setup_time": r.total_setup_time,
                "down_time": r.total_down_time,
                "idle_time": r.total_idle_time,
            }

        avg_utilization = 0
        if self.current_time > 0:
            avg_utilization = round(
                sum(r.total_roasting_time for r in self.roasters.values())
                / (self.current_time * 5) * 100, 1
            )

        return {
            "shift_duration": self.current_time,
            "total_psc_batches": total_psc,
            "psc_L1": self.silos["L1"].total_produced,
            "psc_L2": self.silos["L2"].total_produced,
            "mto_ndg_completed": self.mto_jobs["NDG"]["completed"],
            "mto_ndg_total": self.mto_jobs["NDG"]["completed"] + self.mto_jobs["NDG"]["remaining"],
            "mto_busta_completed": self.mto_jobs["Busta"]["completed"],
            "mto_busta_total": self.mto_jobs["Busta"]["completed"] + self.mto_jobs["Busta"]["remaining"],
            "mto_tardiness": mto_tardiness,
            "stockout_events": total_stockout_events,
            "stockout_duration": total_stockout_duration,
            "roaster_utilization": utilization,
            "avg_utilization": avg_utilization,
            "disruption_log": self.disruptions.get_event_log(),
            "rc_history": {
                "L1": self.rc_history["L1"],
                "L2": self.rc_history["L2"],
            },
        }

    def get_state(self):
        """Get full simulation state for the frontend."""
        return {
            "time": self.current_time,
            "running": self.running,
            "finished": self.finished or self.current_time >= MAX_SHIFT_TIME,
            "auto_mode": self.auto_mode,
            "speed": self.speed,
            "roasters": {
                rid: r.to_dict(self.current_time)
                for rid, r in self.roasters.items()
            },
            "silos": {
                lid: s.to_dict() for lid, s in self.silos.items()
            },
            "pipelines": {
                line: {
                    "busy": self.pipeline_busy[line],
                    "roaster": self.pipeline_batch_roaster[line],
                }
                for line in ["L1", "L2"]
            },
            "mto_jobs": {
                k: {
                    "remaining": v["remaining"],
                    "completed": v["completed"],
                    "completed_time": v["completed_time"],
                }
                for k, v in self.mto_jobs.items()
            },
            "mto_due_date": self.mto_due_date,
            "rc_history": {
                "L1": self.rc_history["L1"],
                "L2": self.rc_history["L2"],
            },
            "recent_events": self.event_log[-20:],
            "decision_log": self.decision_log[-40:],
            "disruptions": self.disruptions.to_dict(self.current_time),
            "gantt": self._get_gantt_data(),
        }

    def _get_gantt_data(self):
        """Get Gantt chart data from roaster histories."""
        gantt = {}
        for rid, roaster in self.roasters.items():
            gantt[rid] = roaster.history
        return gantt
