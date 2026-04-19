"""Universal reactive-strategy dashboard for Nestle Tri An roasting simulation.

Supports Dispatching, Q-Learning, and Manual play modes with live Gantt,
RC/GC history, KPI, event log, and explicit manual restock decisions.
Extends the patterns from env/env_GUI_playground.py without modifying
any env/ code.

Usage:
    python Reactive_GUI.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Callable, Optional

import sys

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from env.data_bridge import get_sim_params
from dispatch.dispatching_heuristic import DispatchingHeuristic
from env.export import export_run
from env.kpi_tracker import KPITracker
from env.simulation_engine import SimulationEngine
from env.simulation_state import BatchRecord, UPSEvent
from env.ups_generator import generate_ups_events


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TITLE = "Nestle Tri An \u2014 Reactive Strategy Dashboard"

COLORS = {
    "bg": "#F7F8FA", "panel": "#FFFFFF", "border": "#D7DCE2",
    "text": "#263238", "muted": "#607D8B",
    "idle": "#FFFFFF", "running": "#C8E6C9", "setup": "#FFF9C4", "down": "#FFCDD2",
    "planned": "#E0E0E0",
    "psc": "#2196F3", "ndg": "#FF9800", "busta": "#4CAF50",
    "setup_bar": "#FFC107", "down_bar": "#F44336", "planned_bar": "#9E9E9E",
    "pipeline_l1": "#1565C0", "pipeline_l2": "#C62828",
    "safety": "#FF5722", "good": "#2E7D32", "warn": "#F9A825", "bad": "#C62828",
    "neutral": "#ECEFF1",
}

ROASTER_ROWS = ["R1", "R2", "R3", "R4", "R5", "P.L1", "P.L2"]

STATUS_BADGE = {
    "IDLE": ("IDLE", COLORS["muted"]),
    "RUNNING": ("RUNNING", COLORS["good"]),
    "SETUP": ("SETUP", COLORS["warn"]),
    "DOWN": ("DOWN", COLORS["bad"]),
}

EVENT_TAGS = {
    "batch_start": {"foreground": "#212121"},
    "batch_complete": {"foreground": "#1B5E20"},
    "setup_start": {"foreground": "#8D6E63"},
    "setup_complete": {"foreground": "#B28704"},
    "ups": {"foreground": "#B71C1C", "font": ("Segoe UI", 9, "bold")},
    "recovery": {"foreground": "#E65100"},
    "stockout": {"foreground": "#B71C1C", "font": ("Segoe UI", 9, "bold")},
    "decision": {"foreground": "#0D47A1"},
    "mto": {"foreground": "#6A1B9A"},
    "restock": {"foreground": "#7B1FA2"},
    "info": {"foreground": "#455A64"},
}

STRATEGY_OPTIONS = (
    "Dispatching",
    "Q-Learning",
    "MaskablePPO",
    "Manual",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LogEntry:
    slot: int
    kind: str
    text: str


# ---------------------------------------------------------------------------
# Tooltip
# ---------------------------------------------------------------------------

class HoverTip:
    def __init__(self, root: tk.Misc):
        self.root = root
        self._tip_window: Optional[tk.Toplevel] = None

    def show(self, text: str, x: int, y: int) -> None:
        self.hide()
        self._tip_window = tk.Toplevel(self.root)
        self._tip_window.wm_overrideredirect(True)
        self._tip_window.wm_attributes("-topmost", True)
        self._tip_window.configure(bg="#FFFDE7")
        tk.Label(
            self._tip_window, text=text, justify="left", bg="#FFFDE7",
            fg=COLORS["text"], relief="solid", borderwidth=1,
            padx=6, pady=4, font=("Segoe UI", 9),
        ).pack()
        self._tip_window.geometry(f"+{x + 14}+{y + 14}")

    def hide(self) -> None:
        if self._tip_window is not None:
            self._tip_window.destroy()
        self._tip_window = None


# ---------------------------------------------------------------------------
# Manual decision dialog
# ---------------------------------------------------------------------------

class ManualDecisionDialog(tk.Toplevel):
    def __init__(self, master, params, engine, state, roaster_id, auto_helper):
        super().__init__(master)
        self.params = params
        self.engine = engine
        self.state = state
        self.roaster_id = roaster_id
        self.auto_helper = auto_helper
        self.result: tuple | None = None
        self._action_map: dict[str, tuple] = {}
        self._ordered_ids: list[str] = []
        self._button_state: dict[str, str] = {}

        self.title(f"Decision: {roaster_id} at t={state.t}")
        self.configure(bg=COLORS["panel"])
        self.resizable(False, False)
        self.transient(master.winfo_toplevel())
        self.grab_set()

        mask = engine._compute_action_mask(state, roaster_id)
        self._build(mask)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _: self._on_cancel())
        self.bind("<Return>", lambda _: self._apply())
        for i in range(1, 6):
            self.bind(str(i), lambda _, idx=i: self._select_index(idx - 1))
        self.wait_visibility()
        self.focus_set()

    def _select_index(self, idx):
        if 0 <= idx < len(self._ordered_ids):
            key = self._ordered_ids[idx]
            if self._button_state.get(key) != "disabled":
                self.choice_var.set(key)

    def _build(self, mask):
        body = tk.Frame(self, bg=COLORS["panel"], padx=16, pady=14)
        body.pack(fill="both", expand=True)

        tk.Label(body, text=f"Decision: {self.roaster_id} at t={self.state.t}",
                 bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 11, "bold"), anchor="w").pack(anchor="w")

        ctx = (f"last_sku={self.state.last_sku[self.roaster_id]}  "
               f"pipeline={self.params['R_pipe'][self.roaster_id]} "
               f"({'FREE' if self.state.pipeline_busy[self.params['R_pipe'][self.roaster_id]] == 0 else 'BUSY'})")
        tk.Label(body, text=ctx, bg=COLORS["panel"], fg=COLORS["muted"],
                 font=("Segoe UI", 9), anchor="w").pack(anchor="w", pady=(2, 4))

        max_rc = int(self.params["max_rc"])
        rc_ctx = (f"RC: L1={self.state.rc_stock['L1']}/{max_rc}  "
                  f"L2={self.state.rc_stock['L2']}/{max_rc}  "
                  f"MTO: J1={self.state.mto_remaining.get('J1', 0)}  J2={self.state.mto_remaining.get('J2', 0)}")
        tk.Label(body, text=rc_ctx, bg=COLORS["panel"], fg=COLORS["muted"],
                 font=("Segoe UI", 9), anchor="w").pack(anchor="w", pady=(0, 2))

        gc_parts = []
        for pair, stock in sorted(self.state.gc_stock.items()):
            gc_parts.append(f"{pair[0]}_{pair[1]}={stock}")
        gc_ctx = f"GC: {', '.join(gc_parts)}" if gc_parts else "GC: n/a"
        rst_ctx = f"  Restock: {'BUSY' if self.state.restock_busy > 0 else 'IDLE'}"
        tk.Label(body, text=gc_ctx + rst_ctx, bg=COLORS["panel"], fg=COLORS["muted"],
                 font=("Segoe UI", 9), anchor="w").pack(anchor="w", pady=(0, 8))

        opts = tk.Frame(body, bg=COLORS["panel"])
        opts.pack(fill="x", expand=True, pady=(2, 10))

        self.choice_var = tk.StringVar(value="")
        ordered = self._ordered_actions(mask)
        for i, action in enumerate(ordered, start=1):
            key = f"opt{i}"
            self._action_map[key] = action
            self._ordered_ids.append(key)
            valid = mask.get(action, False)
            label = self._describe(action)
            if action != ("WAIT",) and self.engine._action_requires_setup(self.state, self.roaster_id, action[0]):
                label += f"  [setup {self.params['sigma']}]"
            btn = ttk.Radiobutton(opts, text=f"{i}. {label}", value=key,
                                  variable=self.choice_var,
                                  state="normal" if valid else "disabled")
            btn.pack(anchor="w", pady=2)
            self._button_state[key] = "normal" if valid else "disabled"
            if valid and not self.choice_var.get():
                self.choice_var.set(key)

        row = tk.Frame(body, bg=COLORS["panel"])
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text="Apply", command=self._apply).pack(side="left")
        ttk.Button(row, text="Auto (heuristic)", command=self._auto).pack(side="left", padx=8)
        ttk.Button(row, text="Cancel", command=self._on_cancel).pack(side="right")

    def _ordered_actions(self, mask):
        ordered = []
        for sku in ("PSC", "NDG", "BUSTA"):
            for a in mask:
                if a == ("WAIT",) or a[0] != sku:
                    continue
                ordered.append(a)
        ordered.append(("WAIT",))
        return ordered

    def _describe(self, action):
        if action == ("WAIT",):
            return "WAIT"
        if action[0] == "PSC":
            return f"PSC -> {action[1]}"
        sku = action[0]
        for jid, js in self.params["job_sku"].items():
            if js == sku:
                return f"{sku} ({jid}: {self.state.mto_remaining.get(jid, 0)} rem)"
        return sku

    def _apply(self):
        self.result = self._action_map.get(self.choice_var.get(), ("WAIT",))
        self.destroy()

    def _auto(self):
        self.result = self.auto_helper.decide(self.state, self.roaster_id)
        self.destroy()

    def _on_cancel(self):
        self.result = ("WAIT",)
        self.destroy()


class ManualRestockDialog(tk.Toplevel):
    def __init__(self, master, params, engine, state, auto_helper):
        super().__init__(master)
        self.params = params
        self.engine = engine
        self.state = state
        self.auto_helper = auto_helper
        self.result: tuple | None = None
        self._action_map: dict[str, tuple] = {}
        self._ordered_ids: list[str] = []
        self._button_state: dict[str, str] = {}

        self.title(f"Restock decision at t={state.t}")
        self.configure(bg=COLORS["panel"])
        self.resizable(False, False)
        self.transient(master.winfo_toplevel())
        self.grab_set()

        feasible = []
        for raw_pair in params.get("feasible_gc_pairs", []):
            line_id, sku = raw_pair
            if engine.can_start_restock(state, line_id, sku):
                feasible.append(("START_RESTOCK", line_id, sku))
        self._build(feasible)
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.bind("<Escape>", lambda _: self._on_cancel())
        self.bind("<Return>", lambda _: self._apply())
        for i in range(1, 6):
            self.bind(str(i), lambda _, idx=i: self._select_index(idx - 1))
        self.wait_visibility()
        self.focus_set()

    def _select_index(self, idx):
        if 0 <= idx < len(self._ordered_ids):
            key = self._ordered_ids[idx]
            if self._button_state.get(key) != "disabled":
                self.choice_var.set(key)

    def _build(self, feasible_actions):
        body = tk.Frame(self, bg=COLORS["panel"], padx=16, pady=14)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text=f"Restock decision at t={self.state.t}",
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        ).pack(anchor="w")

        tk.Label(
            body,
            text=f"Global restock station: {'BUSY' if self.state.restock_busy > 0 else 'FREE'}",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(anchor="w", pady=(2, 2))

        gc_parts = []
        for pair, stock in sorted(self.state.gc_stock.items()):
            cap = self.params.get("gc_capacity", {}).get(pair, "?")
            gc_parts.append(f"{pair[0]}_{pair[1]}={stock}/{cap}")
        tk.Label(
            body,
            text="GC: " + ", ".join(gc_parts) if gc_parts else "GC: n/a",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
            font=("Segoe UI", 9),
            anchor="w",
        ).pack(anchor="w", pady=(0, 8))

        opts = tk.Frame(body, bg=COLORS["panel"])
        opts.pack(fill="x", expand=True, pady=(2, 10))

        self.choice_var = tk.StringVar(value="")
        ordered = sorted(feasible_actions, key=lambda item: (item[1], item[2]))
        ordered.append(("WAIT",))
        for i, action in enumerate(ordered, start=1):
            key = f"opt{i}"
            self._action_map[key] = action
            self._ordered_ids.append(key)
            valid = action == ("WAIT",) or self.engine.can_start_restock(self.state, action[1], action[2])
            btn = ttk.Radiobutton(
                opts,
                text=f"{i}. {self._describe(action)}",
                value=key,
                variable=self.choice_var,
                state="normal" if valid else "disabled",
            )
            btn.pack(anchor="w", pady=2)
            self._button_state[key] = "normal" if valid else "disabled"
            if valid and not self.choice_var.get():
                self.choice_var.set(key)

        row = tk.Frame(body, bg=COLORS["panel"])
        row.pack(fill="x", pady=(12, 0))
        ttk.Button(row, text="Apply", command=self._apply).pack(side="left")
        ttk.Button(row, text="Auto (heuristic)", command=self._auto).pack(side="left", padx=8)
        ttk.Button(row, text="Cancel", command=self._on_cancel).pack(side="right")

    @staticmethod
    def _describe(action):
        if action == ("WAIT",):
            return "WAIT"
        return f"START_RESTOCK {action[1]} {action[2]}"

    def _apply(self):
        self.result = self._action_map.get(self.choice_var.get(), ("WAIT",))
        self.destroy()

    def _auto(self):
        self.result = self.auto_helper.decide_restock(self.state)
        self.destroy()

    def _on_cancel(self):
        self.result = ("WAIT",)
        self.destroy()


# ---------------------------------------------------------------------------
# Steppable engine wrapper
# ---------------------------------------------------------------------------

class SteppableEngine:
    """Single-slot wrapper around SimulationEngine for GUI control."""

    def __init__(self, params, strategy_mode, manual_callback=None, manual_restock_callback=None, log_callback=None):
        self.params = params
        self.strategy_mode = strategy_mode
        self.manual_callback = manual_callback
        self.manual_restock_callback = manual_restock_callback
        self.log_callback = log_callback
        self.dispatch_helper = DispatchingHeuristic(params)

        self.engine = SimulationEngine(params)
        self.state = self.engine._initialize_state()
        self.kpi = self.engine._make_kpi_tracker()
        self.slot_index = 0
        self.strategy = self._make_strategy(strategy_mode)
        self.ups_events: list[UPSEvent] = []
        self.ups_by_time: dict[int, list[UPSEvent]] = defaultdict(list)

        self.rc_history = {lid: [int(params["rc_init"][lid])] for lid in params["lines"]}
        self.gc_history = {
            f"{k[0]}_{k[1]}": [int(v)]
            for k, v in params.get("gc_init", {}).items()
        }
        self.restock_history: list[dict] = []
        self.setup_history = {rid: [] for rid in params["roasters"]}
        self.active_setup = {rid: None for rid in params["roasters"]}
        self.down_history = {rid: [] for rid in params["roasters"]}
        self.active_down = {rid: None for rid in params["roasters"]}
        self.pipeline_history = {lid: [] for lid in params["lines"]}
        self.active_pipeline = {lid: None for lid in params["lines"]}
        self.mto_last_completion = {jid: {"roaster": "-", "time": None} for jid in params["jobs"]}
        self.completed_mto_counts = {jid: 0 for jid in params["jobs"]}

    def _make_strategy(self, mode: str):
        if mode == "Dispatching":
            return DispatchingHeuristic(self.params)
        if mode == "Q-Learning":
            try:
                from q_learning.q_strategy import QStrategy
                q_path = _PROJECT_ROOT / "q_learning" / "q_table.pkl"
                if q_path.exists():
                    return QStrategy(self.params, q_table_path=str(q_path))
                else:
                    if self.log_callback:
                        self.log_callback("info",
                                          "t=0: Q-table not found — falling back to empty Q-table (random)")
                    return QStrategy(self.params)
            except Exception as exc:
                if self.log_callback:
                    self.log_callback("info", f"t=0: Q-Learning load error: {exc}")
                return DispatchingHeuristic(self.params)
        if mode == "MaskablePPO":
            try:
                from PPOmask.Engine.ppo_strategy import PPOStrategy
                from PPOmask.Engine.data_loader import load_data
                ppo_data = load_data()
                return PPOStrategy.load(data=ppo_data, deterministic=True)
            except Exception as exc:
                if self.log_callback:
                    self.log_callback("info", f"t=0: MaskablePPO load error: {exc}")
                return DispatchingHeuristic(self.params)
        return None  # Manual

    @property
    def finished(self):
        return self.slot_index >= int(self.params["SL"])

    def set_ups_events(self, events):
        self.ups_events = sorted(
            [UPSEvent(t=int(e.t), roaster_id=e.roaster_id, duration=int(e.duration)) for e in events],
            key=lambda e: (e.t, e.roaster_id, e.duration),
        )
        self.ups_by_time = defaultdict(list)
        for e in self.ups_events:
            self.ups_by_time[int(e.t)].append(e)

    def add_ups_event(self, event):
        self.ups_events.append(event)
        self.ups_events.sort(key=lambda e: (e.t, e.roaster_id, e.duration))
        self.ups_by_time[int(event.t)].append(event)

    def log(self, kind, text):
        if self.log_callback:
            self.log_callback(kind, text)

    # -- History helpers (same as env_GUI_playground.py) --

    def _finalize_setup(self, rid, end_time):
        a = self.active_setup.get(rid)
        if a is None:
            return
        self.setup_history[rid].append({"start": a["start"], "end": max(a["start"], end_time), "sku": a["sku"]})
        self.active_setup[rid] = None

    def _finalize_down(self, rid, end_time):
        a = self.active_down.get(rid)
        if a is None:
            return
        self.down_history[rid].append({"start": a["start"], "end": max(a["start"] + 1, end_time), "kind": a["kind"]})
        self.active_down[rid] = None

    def _finalize_pipeline(self, lid, end_time):
        a = self.active_pipeline.get(lid)
        if a is None:
            return
        self.pipeline_history[lid].append({"start": a["start"], "end": max(a["start"] + 1, end_time), "batch": a["batch"]})
        self.active_pipeline[lid] = None

    def _ensure_down_started(self, rid, kind="UPS"):
        if self.active_down.get(rid) is None:
            self.active_down[rid] = {"start": int(self.state.t), "kind": kind}

    def _record_timer_transitions(self, before_status, before_batch, before_last_sku):
        for rid in self.params["roasters"]:
            old_s = before_status[rid]
            new_s = self.state.status[rid]
            old_b = before_batch[rid]
            if old_s == "RUNNING" and new_s == "IDLE" and old_b is not None:
                if old_b.is_mto:
                    jid = old_b.batch_id[0]
                    self.completed_mto_counts[jid] += 1
                    self.mto_last_completion[jid] = {"roaster": rid, "time": old_b.end}
                    self.log("mto",
                             f"t={self.state.t}: {rid} completed {old_b.sku} "
                             f"{old_b.batch_id} - MTO {jid}: "
                             f"{self.completed_mto_counts[jid]}/{self.params['job_batches'][jid]} done")
                else:
                    rc_l = old_b.output_line or "-"
                    self.log("batch_complete",
                             f"t={self.state.t}: {rid} completed {old_b.sku} {old_b.batch_id} -> {rc_l}")
            elif old_s == "SETUP" and new_s == "IDLE":
                self._finalize_setup(rid, int(self.state.t))
                self.log("setup_complete",
                         f"t={self.state.t}: {rid} SETUP complete -> last_sku={self.state.last_sku[rid]}")
            elif old_s == "DOWN" and new_s == "IDLE":
                self._finalize_down(rid, int(self.state.t) + 1)
                self.log("recovery", f"t={self.state.t}: {rid} recovered from downtime")

    def _record_pipeline_releases(self, before_busy):
        for lid in self.params["lines"]:
            if before_busy[lid] > 0 and self.state.pipeline_busy[lid] == 0:
                self._finalize_pipeline(lid, int(self.state.t))

    def _record_decision_changes(self, before_status, before_batch):
        for rid in self.params["roasters"]:
            old_s = before_status[rid]
            new_s = self.state.status[rid]
            if old_s == "IDLE" and new_s == "SETUP":
                sku = self.state.setup_target_sku[rid]
                self.active_setup[rid] = {"start": int(self.state.t), "sku": sku}
                self.log("setup_start",
                         f"t={self.state.t}: {rid} entering SETUP for {sku} ({self.params['sigma']} min)")
            elif old_s == "IDLE" and new_s == "RUNNING":
                batch = self.state.current_batch[rid]
                if batch is None:
                    continue
                pl = self.params["R_pipe"][rid]
                self.active_pipeline[pl] = {"start": int(self.state.t), "batch": batch}
                lt = f" -> {batch.output_line}" if batch.output_line else ""
                kind = "mto" if batch.is_mto else "batch_start"
                self.log(kind,
                         f"t={self.state.t}: {rid} started {batch.sku} batch {batch.batch_id}"
                         f"{lt} - pipeline {pl} busy {self.params['DC']} sl")

    def _process_manual_decisions(self):
        for rid in self.params["roasters"]:
            if self.state.status[rid] != "IDLE" or not self.state.needs_decision[rid]:
                continue
            mask = self.engine._compute_action_mask(self.state, rid)
            action = ("WAIT",)
            if self.manual_callback:
                action = self.manual_callback(self.state, rid, mask)
            if not mask.get(action, False):
                action = ("WAIT",)
            self.log("decision", f"t={self.state.t}: manual {rid} -> {action}")
            self.engine._apply_action(self.state, rid, action, self.kpi)

    def _process_manual_restock_decision(self):
        if self.state.restock_busy > 0:
            return
        feasible = []
        for line_id, sku in self.params.get("feasible_gc_pairs", []):
            if self.engine.can_start_restock(self.state, line_id, sku):
                feasible.append(("START_RESTOCK", line_id, sku))
        if not feasible:
            return

        heuristic_action = self.dispatch_helper.decide_restock(self.state)
        urgent = heuristic_action != ("WAIT",) or any(
            self.state.gc_stock.get(pair, 0) <= 1 for pair in self.state.gc_stock
        )
        if not urgent:
            return

        action = ("WAIT",)
        if self.manual_restock_callback:
            action = self.manual_restock_callback(self.state, feasible)
        if len(action) == 3 and action[0] == "START_RESTOCK":
            _, line_id, sku = action
            if self.engine.can_start_restock(self.state, line_id, sku):
                self.engine._start_restock(self.state, line_id, sku, self.kpi)
                rst = self.state.active_restock
                if rst is not None:
                    self.log(
                        "restock",
                        f"t={self.state.t}: manual restock started {rst.line_id} {rst.sku} "
                        f"[{rst.start},{rst.end}) +{rst.qty} GC at completion",
                    )
                return
        self.log("decision", f"t={self.state.t}: manual restock -> {action}")

    # -- Step --

    def step_one_slot(self):
        if self.finished:
            return False

        self.state.t = int(self.slot_index)

        for event in self.ups_by_time.get(self.slot_index, []):
            prior_st = self.state.status[event.roaster_id]
            prior_b = self.state.current_batch[event.roaster_id]
            prior_busy = dict(self.state.pipeline_busy)
            self.engine._process_ups(self.state, event, self.strategy, self.kpi)
            self._ensure_down_started(event.roaster_id)
            if prior_st == "RUNNING" and prior_b is not None:
                self.log("ups",
                         f"t={self.state.t}: UPS on {event.roaster_id} ({event.duration}m) "
                         f"- cancelled {prior_b.sku} {prior_b.batch_id}")
            else:
                self.log("ups", f"t={self.state.t}: UPS on {event.roaster_id} ({event.duration}m)")
            self._record_pipeline_releases(prior_busy)

        bs = dict(self.state.status)
        bb = dict(self.state.current_batch)
        bl = dict(self.state.last_sku)
        self.engine._step_roaster_timers(self.state, self.kpi)
        self._record_timer_transitions(bs, bb, bl)

        bbusy = dict(self.state.pipeline_busy)
        prev_restock_count = len(self.state.completed_restocks)
        self.engine._step_pipeline_and_restock_timers(self.state, self.kpi)
        self._record_pipeline_releases(bbusy)
        if len(self.state.completed_restocks) > prev_restock_count:
            rst = self.state.completed_restocks[-1]
            self.restock_history.append({
                "line_id": rst.line_id, "sku": rst.sku,
                "start": rst.start, "end": rst.end, "qty": rst.qty,
            })
            self.log("restock",
                     f"t={self.state.t}: restock completed {rst.line_id} {rst.sku} "
                     f"+{rst.qty} GC [{rst.start},{rst.end})")

        bso = dict(self.kpi.stockout_events)
        self.engine._process_consumption_events(self.state, self.kpi)
        for lid in self.params["lines"]:
            if self.kpi.stockout_events[lid] > bso[lid]:
                self.log("stockout", f"t={self.state.t}: stockout on {lid} - RC={self.state.rc_stock[lid]}")

        self.engine._track_stockout_duration(self.state, self.kpi)
        self.engine._accrue_idle_penalties(self.state, self.kpi)

        if self.strategy_mode == "Manual":
            self._process_manual_restock_decision()
        else:
            prev_restock = self.state.active_restock
            self.engine._process_restock_decision_point(self.state, self.strategy, self.kpi)
            if self.state.active_restock is not None and self.state.active_restock is not prev_restock:
                rst = self.state.active_restock
                self.log("restock",
                         f"t={self.state.t}: restock started {rst.line_id} {rst.sku} "
                         f"[{rst.start},{rst.end}) +{rst.qty} GC at completion")

        bs2 = dict(self.state.status)
        bb2 = dict(self.state.current_batch)
        if self.strategy_mode == "Manual":
            self._process_manual_decisions()
        else:
            self.engine._process_decision_points(self.state, self.strategy, self.kpi)
        self._record_decision_changes(bs2, bb2)

        self.state.trace.append(self.engine.get_state_snapshot(self.state))
        for lid in self.params["lines"]:
            self.rc_history[lid].append(int(self.state.rc_stock[lid]))
        for pair, stock in self.state.gc_stock.items():
            key = f"{pair[0]}_{pair[1]}"
            if key in self.gc_history:
                self.gc_history[key].append(int(stock))

        self.slot_index += 1
        return True

    def run_to_end(self):
        while self.step_one_slot():
            pass


# ---------------------------------------------------------------------------
# Dashboard application
# ---------------------------------------------------------------------------

class DashboardApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(TITLE)
        root.geometry("1440x880")
        root.minsize(1200, 780)
        root.configure(bg=COLORS["bg"])

        self.params = get_sim_params()
        self.tooltip = HoverTip(root)

        self.strategy_var = tk.StringVar(value=STRATEGY_OPTIONS[0])
        self.speed_scale_var = tk.DoubleVar(value=58.0)
        self.speed_text_var = tk.StringVar(value="")
        self.slot_text_var = tk.StringVar(value=f"Slot: 000/{self.params['SL']}")
        self.banner_var = tk.StringVar(value="Ready")
        self.pipeline_text_var = {lid: tk.StringVar(value=f"Pipeline {lid}: FREE") for lid in self.params["lines"]}
        self.progress_var = tk.DoubleVar(value=0.0)
        self.log_filter_all = tk.BooleanVar(value=True)
        self.log_filter_ups = tk.BooleanVar(value=False)
        self.log_filter_mto = tk.BooleanVar(value=False)
        self.log_filter_stockout = tk.BooleanVar(value=False)
        self.log_filter_decision = tk.BooleanVar(value=False)

        self.auto_running = False
        self.current_job: Optional[threading.Thread] = None
        self.current_job_done: Optional[Callable] = None
        self.current_job_error: Optional[BaseException] = None
        self.current_job_result = None
        self.busy_label_var = tk.StringVar(value="")
        self.log_entries: list[LogEntry] = []
        self.preset_ups_events: list[UPSEvent] = []

        self.roaster_boxes: dict[str, dict] = {}
        self.kpi_value_labels: dict[str, tk.Label] = {}
        self.mto_widgets: dict[str, dict] = {}
        self.sparkline_width = 340
        self.sparkline_height = 72

        self._build_ui()
        self._bind_shortcuts()
        self._update_speed_text()
        self.reset_simulation(initial=True)

    # ---- UI construction ----

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(4, weight=1)
        self._build_toolbar()
        self._build_main_row()
        self._build_pipeline_bar()
        self._build_mto_panel()
        self._build_event_log()

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=COLORS["panel"], bd=1, relief="solid")
        bar.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 6))
        for i in range(15):
            bar.columnconfigure(i, weight=0)
        bar.columnconfigure(12, weight=1)

        ttk.Label(bar, text="Strategy").grid(row=0, column=0, padx=(10, 4), pady=8)
        self.strategy_combo = ttk.Combobox(bar, state="readonly", values=STRATEGY_OPTIONS,
                                           textvariable=self.strategy_var, width=18)
        self.strategy_combo.grid(row=0, column=1, padx=(0, 12), pady=8)

        self.step_button = ttk.Button(bar, text="Step", command=self.step_once, width=9)
        self.step_button.grid(row=0, column=2, padx=2, pady=8)
        self.auto_button = ttk.Button(bar, text="Auto", command=self.toggle_auto, width=9)
        self.auto_button.grid(row=0, column=3, padx=2, pady=8)
        self.full_button = ttk.Button(bar, text="Full", command=self.run_full, width=9)
        self.full_button.grid(row=0, column=4, padx=2, pady=8)
        self.reset_button = ttk.Button(bar, text="Reset", command=self.reset_simulation, width=9)
        self.reset_button.grid(row=0, column=5, padx=2, pady=8)

        ttk.Label(bar, text="Speed").grid(row=0, column=6, padx=(10, 4))
        self.speed_scale = ttk.Scale(bar, from_=0, to=100, variable=self.speed_scale_var,
                                     command=lambda _: self._update_speed_text(), length=150)
        self.speed_scale.grid(row=0, column=7, padx=(0, 4), pady=8)
        ttk.Label(bar, textvariable=self.speed_text_var, width=11).grid(row=0, column=8, padx=(0, 10))

        ttk.Button(bar, text="Inject UPS", command=self.open_inject_ups_dialog).grid(row=0, column=9, padx=2)
        ttk.Button(bar, text="UPS Preset", command=self.open_ups_preset_dialog).grid(row=0, column=10, padx=2)
        ttk.Button(bar, text="Export", command=self.export_current_run).grid(row=0, column=11, padx=2)
        ttk.Button(bar, text="Compare", command=self.open_compare_dialog).grid(row=0, column=12, padx=2, sticky="w")

        sf = tk.Frame(bar, bg=COLORS["panel"])
        sf.grid(row=0, column=14, padx=(12, 10), pady=6, sticky="e")
        tk.Label(sf, textvariable=self.slot_text_var, bg=COLORS["panel"], fg=COLORS["text"],
                 font=("Segoe UI", 10, "bold")).pack(anchor="e")
        self.progress = ttk.Progressbar(sf, orient="horizontal", mode="determinate",
                                        maximum=self.params["SL"], variable=self.progress_var, length=160)
        self.progress.pack(anchor="e", pady=(4, 0))
        self.banner_label = tk.Label(sf, textvariable=self.banner_var, bg=COLORS["panel"],
                                     fg=COLORS["muted"], font=("Segoe UI", 9))
        self.banner_label.pack(anchor="e", pady=(2, 0))

    def _build_main_row(self):
        c = tk.Frame(self.root, bg=COLORS["bg"])
        c.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 6))
        c.columnconfigure(0, weight=3)
        c.columnconfigure(1, weight=6)
        c.columnconfigure(2, weight=3)
        c.rowconfigure(0, weight=1)
        self._build_roaster_panel(c)
        self._build_center_panel(c)
        self._build_kpi_panel(c)

    def _build_roaster_panel(self, parent):
        panel = tk.LabelFrame(parent, text="Roasters", bg=COLORS["panel"], fg=COLORS["text"])
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        panel.columnconfigure(0, weight=1)
        for idx, rid in enumerate(self.params["roasters"]):
            frame = tk.Frame(panel, bg=COLORS["idle"], bd=1, relief="solid",
                             highlightthickness=0, cursor="hand2")
            frame.grid(row=idx, column=0, sticky="ew", padx=8, pady=6)
            frame.columnconfigure(0, weight=1)
            head = tk.Frame(frame, bg=COLORS["idle"])
            head.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
            head.columnconfigure(0, weight=1)
            name = tk.Label(head, text=rid, bg=COLORS["idle"], fg=COLORS["text"],
                            font=("Segoe UI", 11, "bold"))
            name.grid(row=0, column=0, sticky="w")
            badge = tk.Label(head, text="IDLE", bg=COLORS["idle"], fg=COLORS["muted"],
                             font=("Segoe UI", 9, "bold"))
            badge.grid(row=0, column=1, sticky="e")
            activity = tk.Label(frame, text="IDLE", bg=COLORS["idle"], fg=COLORS["text"],
                                font=("Segoe UI", 9), justify="left", anchor="w", wraplength=240)
            activity.grid(row=1, column=0, sticky="ew", padx=8)
            detail = tk.Label(frame, text="remaining: 0", bg=COLORS["idle"], fg=COLORS["muted"],
                              font=("Segoe UI", 8), justify="left", anchor="w", wraplength=240)
            detail.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 8))
            self.roaster_boxes[rid] = {"frame": frame, "head": head, "name": name,
                                       "badge": badge, "activity": activity, "detail": detail}
            for w in (frame, head, name, badge, activity, detail):
                w.bind("<Button-1>", lambda _, r=rid: self.show_roaster_details(r))

    def _build_center_panel(self, parent):
        panel = tk.Frame(parent, bg=COLORS["bg"])
        panel.grid(row=0, column=1, sticky="nsew", padx=6)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=4)
        panel.rowconfigure(1, weight=1)
        panel.rowconfigure(2, weight=1)

        gf = tk.LabelFrame(panel, text="Schedule Gantt", bg=COLORS["panel"], fg=COLORS["text"])
        gf.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        gf.columnconfigure(0, weight=1)
        gf.rowconfigure(0, weight=1)
        self.gantt_canvas = tk.Canvas(gf, bg="white", height=320, highlightthickness=0)
        self.gantt_canvas.grid(row=0, column=0, sticky="nsew")

        rf = tk.LabelFrame(panel, text="RC Stock", bg=COLORS["panel"], fg=COLORS["text"])
        rf.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        rf.columnconfigure(0, weight=1)
        self.rc_canvas = tk.Canvas(rf, bg="white", height=80, highlightthickness=0)
        self.rc_canvas.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 2))
        self.sparkline_canvas = tk.Canvas(rf, bg="white", height=self.sparkline_height, highlightthickness=0)
        self.sparkline_canvas.grid(row=1, column=0, sticky="ew", padx=8, pady=(2, 6))

        gcf = tk.LabelFrame(panel, text="GC Silo Inventory", bg=COLORS["panel"], fg=COLORS["text"])
        gcf.grid(row=2, column=0, sticky="nsew")
        gcf.columnconfigure(0, weight=1)
        self.gc_canvas = tk.Canvas(gcf, bg="white", height=100, highlightthickness=0)
        self.gc_canvas.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 6))

    def _build_kpi_panel(self, parent):
        panel = tk.LabelFrame(parent, text="KPI Panel", bg=COLORS["panel"], fg=COLORS["text"])
        panel.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        panel.columnconfigure(0, weight=1)
        rows = [
            ("net_profit", "Net Profit"), ("revenue", "Revenue"), ("costs", "Costs"),
            ("psc", "PSC"), ("ndg", "NDG"), ("busta", "BUSTA"),
            ("tard", "Tardiness"), ("stockout", "Stockout"), ("setup", "Setup"),
            ("restock", "Restocks"), ("gc_stock", "GC Stock"),
            ("idle", "Idle"), ("over", "Over-idle"),
            ("batches", "Batches"), ("cancelled", "Cancelled"),
            ("ups", "UPS fired"), ("compute", "Compute"),
        ]
        for idx, (key, ltxt) in enumerate(rows):
            bold = key == "net_profit"
            tk.Label(panel, text=ltxt, bg=COLORS["panel"], fg=COLORS["muted"], anchor="w",
                     font=("Segoe UI", 10 if bold else 9, "bold" if bold else "normal")
                     ).grid(row=idx * 2, column=0, sticky="ew", padx=12, pady=(10 if idx == 0 else 6, 0))
            v = tk.Label(panel, text="-", bg=COLORS["panel"], fg=COLORS["text"], anchor="w", justify="left",
                         font=("Segoe UI", 16, "bold") if bold else ("Consolas", 10), wraplength=300)
            v.grid(row=idx * 2 + 1, column=0, sticky="ew", padx=12)
            self.kpi_value_labels[key] = v

        self.busy_label = tk.Label(panel, textvariable=self.busy_label_var, bg=COLORS["panel"],
                                   fg=COLORS["bad"], anchor="w", justify="left",
                                   font=("Segoe UI", 10, "bold"), wraplength=300)
        self.busy_label.grid(row=100, column=0, sticky="ew", padx=12, pady=(12, 12))

    def _build_pipeline_bar(self):
        panel = tk.LabelFrame(self.root, text="Pipeline Status", bg=COLORS["panel"], fg=COLORS["text"])
        panel.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 6))
        panel.columnconfigure(0, weight=1)
        panel.columnconfigure(1, weight=1)
        self.pipeline_labels = {}
        for i, lid in enumerate(self.params["lines"]):
            lbl = tk.Label(panel, textvariable=self.pipeline_text_var[lid], bg=COLORS["panel"],
                           fg=COLORS["good"], anchor="w", font=("Segoe UI", 10, "bold"), padx=10, pady=8)
            lbl.grid(row=0, column=i, sticky="ew")
            self.pipeline_labels[lid] = lbl

    def _build_mto_panel(self):
        panel = tk.LabelFrame(self.root, text="MTO Tracker", bg=COLORS["panel"], fg=COLORS["text"])
        panel.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 6))
        panel.columnconfigure(0, weight=1)
        for idx, jid in enumerate(self.params["jobs"]):
            row = tk.Frame(panel, bg=COLORS["panel"])
            row.grid(row=idx, column=0, sticky="ew", padx=10, pady=6)
            row.columnconfigure(1, weight=1)
            tk.Label(row, text=jid, bg=COLORS["panel"], fg=COLORS["text"],
                     font=("Segoe UI", 10, "bold"), width=14, anchor="w").grid(row=0, column=0, sticky="w")
            prog = ttk.Progressbar(row, maximum=self.params["job_batches"][jid], length=350)
            prog.grid(row=0, column=1, sticky="ew", padx=8)
            det = tk.Label(row, text="-", bg=COLORS["panel"], fg=COLORS["muted"],
                           font=("Consolas", 9), anchor="w")
            det.grid(row=0, column=2, sticky="w", padx=(8, 0))
            self.mto_widgets[jid] = {"progress": prog, "detail": det}

    def _build_event_log(self):
        panel = tk.LabelFrame(self.root, text="Event Log", bg=COLORS["panel"], fg=COLORS["text"])
        panel.grid(row=4, column=0, sticky="nsew", padx=10, pady=(0, 10))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        fr = tk.Frame(panel, bg=COLORS["panel"])
        fr.grid(row=0, column=0, sticky="ew", padx=10, pady=(6, 0))
        ttk.Checkbutton(fr, text="All", variable=self.log_filter_all, command=self.refresh_event_log).pack(side="left", padx=(0, 8))
        for txt, var in [("UPS", self.log_filter_ups), ("MTO", self.log_filter_mto),
                         ("Stockout", self.log_filter_stockout), ("Decisions", self.log_filter_decision)]:
            ttk.Checkbutton(fr, text=txt, variable=var, command=self.refresh_event_log).pack(side="left", padx=4)
        self.log_text = scrolledtext.ScrolledText(panel, height=10, wrap="word", bg="white",
                                                  fg=COLORS["text"], font=("Consolas", 9))
        self.log_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=8)
        self.log_text.configure(state="disabled")
        for tag, cfg in EVENT_TAGS.items():
            self.log_text.tag_configure(tag, **cfg)

    # ---- Shortcuts ----

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda _: self.step_once())
        self.root.bind("<Right>", lambda _: self.step_once())
        self.root.bind("<Return>", lambda _: self.toggle_auto())
        self.root.bind("<Escape>", lambda _: self.stop_auto())
        self.root.bind("f", lambda _: self.run_full())
        self.root.bind("F", lambda _: self.run_full())
        self.root.bind("r", lambda _: self.reset_simulation())
        self.root.bind("R", lambda _: self.reset_simulation())

    # ---- Speed ----

    def _speed_slots_per_second(self):
        ratio = float(self.speed_scale_var.get()) / 100.0
        return min(200, max(1, int(round(10 ** (ratio * 2.30103)))))

    def _update_speed_text(self):
        self.speed_text_var.set(f"{self._speed_slots_per_second()} slots/s")

    def _step_delay_ms(self):
        return max(5, int(1000 / self._speed_slots_per_second()))

    # ---- Simulation control ----

    def reset_simulation(self, initial=False):
        self.stop_auto()
        self.busy_label_var.set("")
        self.log_entries = []
        self.current_job = None
        self.current_job_error = None
        self.current_job_result = None
        self.current_job_done = None
        # Reload numeric parameters from Input_data so Reset picks up CSV edits.
        self.params = get_sim_params()
        self.progress.configure(maximum=self.params["SL"])

        self.steppable = SteppableEngine(
            self.params, self.strategy_var.get(),
            manual_callback=self._manual_cb,
            manual_restock_callback=self._manual_restock_cb,
            log_callback=self.add_log_entry,
        )
        self.steppable.set_ups_events(list(self.preset_ups_events))
        self.add_log_entry("info",
                           f"t=0: reset - strategy={self.strategy_var.get()}, UPS={len(self.preset_ups_events)}")
        for e in self.preset_ups_events:
            self.add_log_entry("ups", f"t={e.t}: [scheduled] UPS on {e.roaster_id} for {e.duration}m")
        self.refresh_all()
        if not initial:
            self.banner_var.set("Reset to slot 0")

    def stop_auto(self):
        self.auto_running = False
        self.auto_button.configure(text="Auto")

    def _can_change_strategy(self):
        return self.steppable.slot_index == 0 and not self.auto_running and self.current_job is None

    def step_once(self):
        if self.auto_running or self.current_job is not None or self.steppable.finished:
            return
        self.banner_var.set("Stepping...")
        mode = self.strategy_var.get()
        if mode == "Q-Learning":
            self._start_bg("Solving / stepping...", self._bg_step, self._after_step)
        else:
            self._step_sync()

    def _step_sync(self):
        stepped = self.steppable.step_one_slot()
        self.refresh_all()
        if not stepped or self.steppable.finished:
            self.banner_var.set("SHIFT COMPLETE")
        else:
            self.banner_var.set("Paused")

    def _bg_step(self):
        return self.steppable.step_one_slot()

    def _after_step(self, stepped):
        self.refresh_all()
        if not stepped or self.steppable.finished:
            self.stop_auto()
            self.banner_var.set("SHIFT COMPLETE")
        else:
            self.banner_var.set("Paused")

    def toggle_auto(self):
        if self.current_job is not None:
            return
        if self.auto_running:
            self.stop_auto()
            self.banner_var.set("Paused")
            return
        if self.steppable.finished:
            return
        self.auto_running = True
        self.auto_button.configure(text="Pause")
        self.banner_var.set("Running...")
        self._schedule_auto()

    def _schedule_auto(self):
        if not self.auto_running or self.steppable.finished:
            self.stop_auto()
            self.banner_var.set("SHIFT COMPLETE" if self.steppable.finished else "Paused")
            self.refresh_all()
            return
        mode = self.strategy_var.get()
        if mode == "Q-Learning":
            self._start_bg("Solving / auto-run...", self._bg_step, self._after_auto_async)
        else:
            self._step_sync()
            if self.auto_running:
                self.root.after(self._step_delay_ms(), self._schedule_auto)

    def _after_auto_async(self, stepped):
        self.refresh_all()
        if not self.auto_running or not stepped or self.steppable.finished:
            self.stop_auto()
            self.banner_var.set("SHIFT COMPLETE" if self.steppable.finished else "Paused")
            return
        self.root.after(self._step_delay_ms(), self._schedule_auto)

    def run_full(self):
        if self.current_job is not None or self.auto_running or self.steppable.finished:
            return
        if self.strategy_var.get() == "Manual":
            self.banner_var.set("Full run in manual mode")
            while not self.steppable.finished:
                self._step_sync()
                self.root.update_idletasks()
                self.root.update()
            return
        self.banner_var.set("Running full shift...")
        self._start_bg("Running full shift...", self._bg_full, self._after_full)

    def _bg_full(self):
        self.steppable.run_to_end()
        return True

    def _after_full(self, _):
        self.refresh_all()
        self.banner_var.set("SHIFT COMPLETE")

    # ---- Background jobs ----

    def _start_bg(self, label, func, done_cb):
        if self.current_job is not None:
            return
        self.busy_label_var.set(label)
        self.current_job_error = None
        self.current_job_result = None
        self.current_job_done = done_cb

        def runner():
            try:
                self.current_job_result = func()
            except BaseException as exc:
                self.current_job_error = exc

        self.current_job = threading.Thread(target=runner, daemon=True)
        self.current_job.start()
        self.root.after(60, self._poll_bg)

    def _poll_bg(self):
        job = self.current_job
        if job is None:
            return
        if job.is_alive():
            self.root.after(60, self._poll_bg)
            return
        self.current_job = None
        self.busy_label_var.set("")
        if self.current_job_error is not None:
            err = self.current_job_error
            self.current_job_error = None
            self.stop_auto()
            self.refresh_all()
            messagebox.showerror("Simulation Error", str(err))
            return
        cb = self.current_job_done
        res = self.current_job_result
        self.current_job_done = None
        self.current_job_result = None
        if cb:
            cb(res)

    # ---- Manual callback ----

    def _manual_cb(self, state, roaster_id, _mask):
        dlg = ManualDecisionDialog(self.root, self.params, self.steppable.engine,
                                   state, roaster_id, self.steppable.dispatch_helper)
        self.root.wait_window(dlg)
        return dlg.result or ("WAIT",)

    def _manual_restock_cb(self, state, _feasible_actions):
        dlg = ManualRestockDialog(
            self.root,
            self.params,
            self.steppable.engine,
            state,
            self.steppable.dispatch_helper,
        )
        self.root.wait_window(dlg)
        return dlg.result or ("WAIT",)

    # ---- Logging ----

    def add_log_entry(self, kind, text):
        entry = LogEntry(slot=self.steppable.slot_index if hasattr(self, "steppable") else 0,
                         kind=kind, text=text)
        self.log_entries.append(entry)
        self.log_entries = self.log_entries[-300:]
        self.refresh_event_log()

    def _visible_log_entries(self):
        if self.log_filter_all.get():
            return self.log_entries[-50:]
        allowed = set()
        if self.log_filter_ups.get():
            allowed.update({"ups", "recovery"})
        if self.log_filter_mto.get():
            allowed.add("mto")
        if self.log_filter_stockout.get():
            allowed.add("stockout")
        if self.log_filter_decision.get():
            allowed.add("decision")
        if not allowed:
            return self.log_entries[-50:]
        return [e for e in self.log_entries if e.kind in allowed][-50:]

    def refresh_event_log(self):
        entries = list(reversed(self._visible_log_entries()))
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        for e in entries:
            tag = e.kind if e.kind in EVENT_TAGS else "info"
            self.log_text.insert("end", e.text + "\n", tag)
        self.log_text.configure(state="disabled")

    # ---- Refresh all ----

    def refresh_all(self):
        self.slot_text_var.set(f"Slot: {self.steppable.slot_index:03d}/{self.params['SL']}")
        self.progress_var.set(self.steppable.slot_index)
        if self.steppable.finished:
            self.banner_var.set("SHIFT COMPLETE")
        self.strategy_combo.configure(state="readonly" if self._can_change_strategy() else "disabled")
        self.step_button.configure(state="normal" if not self.auto_running and self.current_job is None else "disabled")
        self.full_button.configure(state="normal" if not self.auto_running and self.current_job is None else "disabled")
        self.reset_button.configure(state="normal" if self.current_job is None else "disabled")
        self._refresh_roaster_boxes()
        self._refresh_gantt()
        self._refresh_rc_display()
        self._refresh_gc_display()
        self._refresh_kpi_panel()
        self._refresh_pipeline_bar()
        self._refresh_mto_panel()
        self.refresh_event_log()

    # ---- Roaster boxes ----

    def _refresh_roaster_boxes(self):
        state = self.steppable.state
        for rid in self.params["roasters"]:
            box = self.roaster_boxes[rid]
            status = state.status[rid]
            color = COLORS["idle"]
            if status == "RUNNING":
                color = COLORS["running"]
            elif status == "SETUP":
                color = COLORS["setup"]
            elif status == "DOWN":
                color = COLORS["down"]
            elif self.steppable.slot_index in self.params["downtime_slots"].get(rid, set()):
                color = COLORS["planned"]
            for k in ("frame", "head", "name", "badge", "activity", "detail"):
                box[k].configure(bg=color)
            bt, bc = STATUS_BADGE.get(status, (status, COLORS["text"]))
            box["badge"].configure(text=bt, fg=bc)
            batch = state.current_batch[rid]
            if status == "RUNNING" and batch:
                act = f"{batch.sku} batch {batch.batch_id}"
                if batch.output_line:
                    act += f" -> {batch.output_line}"
                det = f"remaining: {state.remaining[rid]}   last_sku: {state.last_sku[rid]}"
            elif status == "SETUP":
                act = f"SETUP for {state.setup_target_sku[rid]} (rem={state.remaining[rid]})"
                det = f"last_sku: {state.last_sku[rid]}"
            elif status == "DOWN":
                act = f"DOWN (rem={state.remaining[rid]})"
                det = f"last_sku: {state.last_sku[rid]}"
            else:
                act = f"IDLE (needs_decision={state.needs_decision[rid]})"
                det = f"last_sku: {state.last_sku[rid]}   remaining: {state.remaining[rid]}"
            box["activity"].configure(text=act)
            box["detail"].configure(text=det)

    # ---- Gantt ----

    def _gantt_x(self, slot, width, lm, rm):
        usable = max(20, width - lm - rm)
        return lm + usable * float(slot) / float(self.params["SL"])

    def _bind_tip(self, item_id, text):
        self.gantt_canvas.tag_bind(item_id, "<Enter>",
                                   lambda e, m=text: self.tooltip.show(m, e.x_root, e.y_root))
        self.gantt_canvas.tag_bind(item_id, "<Leave>", lambda _: self.tooltip.hide())

    def _draw_hatched(self, x1, y1, x2, y2, fill):
        self.gantt_canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline=fill)
        for off in range(int(x1) - 20, int(x2) + 20, 8):
            self.gantt_canvas.create_line(off, y2, off + 20, y1, fill="#B0BEC5", width=1)

    @staticmethod
    def _slot_intervals(slots):
        ordered = sorted(int(slot) for slot in slots)
        if not ordered:
            return []
        intervals = []
        start = ordered[0]
        prev = ordered[0]
        for slot in ordered[1:]:
            if slot == prev + 1:
                prev = slot
                continue
            intervals.append((start, prev + 1))
            start = prev = slot
        intervals.append((start, prev + 1))
        return intervals

    def _refresh_gantt(self):
        c = self.gantt_canvas
        c.delete("all")
        w = max(c.winfo_width(), 600)
        h = max(c.winfo_height(), 320)
        lm, rm, tm = 52, 10, 18
        rh = max(24, int((h - tm - 28) / len(ROASTER_ROWS)))

        for idx, rn in enumerate(ROASTER_ROWS):
            y = tm + idx * rh
            c.create_text(8, y + rh / 2, anchor="w", text=rn, fill=COLORS["text"], font=("Segoe UI", 9, "bold"))
            c.create_line(lm, y + rh, w - rm, y + rh, fill="#ECEFF1")

        for tick in range(0, self.params["SL"] + 1, 60):
            x = self._gantt_x(tick, w, lm, rm)
            c.create_line(x, tm - 8, x, h - 8, fill="#ECEFF1")
            c.create_text(x, 8, text=str(tick), fill=COLORS["muted"], font=("Segoe UI", 8))

        color_sku = {"PSC": COLORS["psc"], "NDG": COLORS["ndg"], "BUSTA": COLORS["busta"]}
        ri = {n: i for i, n in enumerate(ROASTER_ROWS)}

        for rid in self.params["roasters"]:
            if rid not in ri:
                continue
            y = tm + ri[rid] * rh + 3
            for start, end in self._slot_intervals(self.params["downtime_slots"].get(rid, set())):
                x1 = self._gantt_x(start, w, lm, rm)
                x2 = self._gantt_x(end, w, lm, rm)
                self._draw_hatched(x1, y, x2, y + rh - 6, COLORS["planned_bar"])

        for rid, intervals in self.steppable.setup_history.items():
            y = tm + ri[rid] * rh + 5
            for it in intervals:
                x1 = self._gantt_x(it["start"], w, lm, rm)
                x2 = self._gantt_x(it["end"], w, lm, rm)
                rect = c.create_rectangle(x1, y, x2, y + rh - 10, fill=COLORS["setup_bar"], outline="#B28704")
                self._bind_tip(rect, f"SETUP {rid} -> {it['sku']} [{it['start']},{it['end']})")

        for rid, intervals in self.steppable.down_history.items():
            y = tm + ri[rid] * rh + 5
            for it in intervals:
                x1 = self._gantt_x(it["start"], w, lm, rm)
                x2 = self._gantt_x(it["end"], w, lm, rm)
                rect = c.create_rectangle(x1, y, x2, y + rh - 10, fill=COLORS["down_bar"], outline="#B71C1C")
                self._bind_tip(rect, f"DOWN {rid} [{it['start']},{it['end']})")

        for batch in self.steppable.state.completed_batches:
            y = tm + ri[batch.roaster] * rh + 3
            x1 = self._gantt_x(batch.start, w, lm, rm)
            x2 = self._gantt_x(batch.end, w, lm, rm)
            rect = c.create_rectangle(x1, y, x2, y + rh - 6, fill=color_sku[batch.sku],
                                      outline="#263238", width=2 if batch.is_mto else 1)
            tip = f"{batch.sku} {batch.batch_id} [{batch.start},{batch.end})"
            if batch.output_line:
                tip += f" -> {batch.output_line}"
            self._bind_tip(rect, tip)

        for rid in self.params["roasters"]:
            batch = self.steppable.state.current_batch[rid]
            if not batch:
                continue
            y = tm + ri[rid] * rh + 3
            x1 = self._gantt_x(batch.start, w, lm, rm)
            x2 = self._gantt_x(self.steppable.slot_index, w, lm, rm)
            rect = c.create_rectangle(x1, y, max(x1 + 2, x2), y + rh - 6,
                                      fill=color_sku[batch.sku], outline="#000", dash=(3, 2), width=2)
            self._bind_tip(rect, f"ACTIVE {batch.sku} {batch.batch_id} from {batch.start}")

        for rid, it in self.steppable.active_setup.items():
            if not it:
                continue
            y = tm + ri[rid] * rh + 5
            x1 = self._gantt_x(it["start"], w, lm, rm)
            x2 = self._gantt_x(self.steppable.slot_index, w, lm, rm)
            c.create_rectangle(x1, y, max(x1 + 2, x2), y + rh - 10,
                               fill=COLORS["setup_bar"], outline="#B28704", dash=(2, 2))

        for rid, it in self.steppable.active_down.items():
            if not it:
                continue
            y = tm + ri[rid] * rh + 5
            x1 = self._gantt_x(it["start"], w, lm, rm)
            x2 = self._gantt_x(self.steppable.slot_index, w, lm, rm)
            c.create_rectangle(x1, y, max(x1 + 2, x2), y + rh - 10,
                               fill=COLORS["down_bar"], outline="#B71C1C", dash=(2, 2))

        for lid in self.params["lines"]:
            rn = f"P.{lid}"
            y = tm + ri[rn] * rh + 5
            fill = COLORS["pipeline_l1"] if lid == "L1" else COLORS["pipeline_l2"]
            for it in self.steppable.pipeline_history[lid]:
                x1 = self._gantt_x(it["start"], w, lm, rm)
                x2 = self._gantt_x(it["end"], w, lm, rm)
                rect = c.create_rectangle(x1, y, x2, y + rh - 10, fill=fill, outline=fill)
                b = it["batch"]
                if hasattr(b, "sku"):
                    self._bind_tip(rect, f"Pipeline {lid}: {b.sku} {b.batch_id} [{it['start']},{it['end']})")
                else:
                    self._bind_tip(rect, f"Pipeline {lid}: [{it['start']},{it['end']})")
            for rst in self.steppable.restock_history:
                if rst["line_id"] != lid:
                    continue
                x1 = self._gantt_x(rst["start"], w, lm, rm)
                x2 = self._gantt_x(rst["end"], w, lm, rm)
                rect = c.create_rectangle(x1, y, x2, y + rh - 10, fill="#7B1FA2", outline="#4A148C")
                self._bind_tip(rect,
                               f"RESTOCK {lid} {rst['sku']} [{rst['start']},{rst['end']}) +{rst['qty']} GC")
            act = self.steppable.active_pipeline.get(lid)
            if act:
                x1 = self._gantt_x(act["start"], w, lm, rm)
                x2 = self._gantt_x(self.steppable.slot_index, w, lm, rm)
                mode = self.steppable.state.pipeline_mode.get(lid, "FREE")
                pfill = "#7B1FA2" if mode == "RESTOCK" else fill
                c.create_rectangle(x1, y, max(x1 + 2, x2), y + rh - 10, fill=pfill, outline=pfill, dash=(2, 2))

        cx = self._gantt_x(self.steppable.slot_index, w, lm, rm)
        c.create_line(cx, tm - 8, cx, h - 8, fill=COLORS["bad"], width=2)

    # ---- RC display ----

    def _rc_bar_color(self, val):
        if val <= 0:
            return "#B71C1C"
        if val < 10:
            return COLORS["bad"]
        if val <= self.params["safety_stock"]:
            return COLORS["warn"]
        return COLORS["good"]

    def _refresh_rc_display(self):
        c = self.rc_canvas
        c.delete("all")
        w = max(c.winfo_width(), 360)
        bw = w - 120
        bh = 24
        x0 = 72
        sx = x0 + bw * (self.params["safety_stock"] / self.params["max_rc"])
        for idx, lid in enumerate(self.params["lines"]):
            val = int(self.steppable.state.rc_stock[lid])
            y0 = 14 + idx * 36
            y1 = y0 + bh
            c.create_text(12, y0 + 12, text=f"{lid}:", anchor="w", fill=COLORS["text"],
                          font=("Segoe UI", 10, "bold"))
            c.create_rectangle(x0, y0, x0 + bw, y1, fill=COLORS["neutral"], outline=COLORS["border"])
            fw = max(0, min(bw, bw * (max(0, val) / self.params["max_rc"])))
            c.create_rectangle(x0, y0, x0 + fw, y1, fill=self._rc_bar_color(val), outline="")
            c.create_line(sx, y0 - 4, sx, y1 + 4, fill=COLORS["safety"], dash=(4, 3))
            label = f"{val}/{self.params['max_rc']} batches"
            if val <= 0:
                label += "  DEPLETED"
            c.create_text(x0 + bw + 8, y0 + 12, text=label, anchor="w", fill=COLORS["text"],
                          font=("Consolas", 9))
        self._refresh_sparkline()

    def _refresh_sparkline(self):
        c = self.sparkline_canvas
        c.delete("all")
        w = max(c.winfo_width(), self.sparkline_width)
        h = self.sparkline_height
        left, right, top, bottom = 20, w - 10, 8, h - 12
        hl = min(200, len(self.steppable.rc_history["L1"]))
        if hl <= 1:
            return
        si = len(self.steppable.rc_history["L1"]) - hl
        v1 = self.steppable.rc_history["L1"][si:]
        v2 = self.steppable.rc_history["L2"][si:]
        c.create_line(left, bottom, right, bottom, fill="#ECEFF1")
        c.create_line(left, top, left, bottom, fill="#ECEFF1")
        sy = bottom - (bottom - top) * (self.params["safety_stock"] / self.params["max_rc"])
        c.create_line(left, sy, right, sy, fill=COLORS["safety"], dash=(4, 3))

        def pts(vals):
            out = []
            for i, v in enumerate(vals):
                x = left + (right - left) * (i / max(1, len(vals) - 1))
                y = bottom - (bottom - top) * (max(0, min(self.params["max_rc"], v)) / self.params["max_rc"])
                out.extend([x, y])
            return out

        c.create_line(*pts(v1), fill=COLORS["pipeline_l1"], width=2, smooth=False)
        c.create_line(*pts(v2), fill=COLORS["pipeline_l2"], width=2, smooth=False)
        c.create_text(right, top + 4, anchor="ne", text="L1", fill=COLORS["pipeline_l1"],
                      font=("Segoe UI", 8, "bold"))
        c.create_text(right, top + 18, anchor="ne", text="L2", fill=COLORS["pipeline_l2"],
                      font=("Segoe UI", 8, "bold"))

    # ---- GC display ----

    def _gc_bar_color(self, val, cap):
        if val <= 0:
            return "#B71C1C"
        ratio = val / max(1, cap)
        if ratio < 0.15:
            return COLORS["bad"]
        if ratio < 0.35:
            return COLORS["warn"]
        return COLORS["good"]

    def _refresh_gc_display(self):
        c = self.gc_canvas
        c.delete("all")
        w = max(c.winfo_width(), 360)
        gc_stock = self.steppable.state.gc_stock
        gc_cap = self.params.get("gc_capacity", {})
        pairs = sorted(gc_stock.keys())
        if not pairs:
            c.create_text(w // 2, 40, text="No GC silos", fill=COLORS["muted"],
                          font=("Segoe UI", 10))
            return

        bw = w - 160
        bh = 18
        x0 = 100
        for idx, pair in enumerate(pairs):
            stock = int(gc_stock[pair])
            cap = int(gc_cap.get(pair, 40))
            label = f"{pair[0]}_{pair[1]}"
            y0 = 8 + idx * (bh + 6)
            y1 = y0 + bh
            c.create_text(8, y0 + bh // 2, text=f"{label}:", anchor="w", fill=COLORS["text"],
                          font=("Segoe UI", 9, "bold"))
            c.create_rectangle(x0, y0, x0 + bw, y1, fill=COLORS["neutral"], outline=COLORS["border"])
            fw = max(0, min(bw, bw * (max(0, stock) / max(1, cap))))
            c.create_rectangle(x0, y0, x0 + fw, y1, fill=self._gc_bar_color(stock, cap), outline="")
            txt = f"{stock}/{cap}"
            if stock <= 0:
                txt += " EMPTY"
            c.create_text(x0 + bw + 8, y0 + bh // 2, text=txt, anchor="w", fill=COLORS["text"],
                          font=("Consolas", 9))

        rst = self.steppable.state.active_restock
        if rst:
            y_rst = 8 + len(pairs) * (bh + 6)
            c.create_text(8, y_rst, text=f"RESTOCK: {rst.line_id} {rst.sku} "
                          f"[{rst.start},{rst.end}) busy={self.steppable.state.restock_busy}",
                          anchor="w", fill="#7B1FA2", font=("Segoe UI", 9, "bold"))

    # ---- KPI panel ----

    def _refresh_kpi_panel(self):
        r = self.steppable.kpi.to_dict()
        net = r["net_profit"]
        self.kpi_value_labels["net_profit"].configure(text=f"${net:,.0f}",
                                                      fg=COLORS["good"] if net >= 0 else COLORS["bad"])
        self.kpi_value_labels["revenue"].configure(text=f"${r['total_revenue']:,.0f}")
        self.kpi_value_labels["costs"].configure(text=f"-${r['total_costs']:,.0f}")
        self.kpi_value_labels["psc"].configure(text=f"{r['psc_count']} batches (${r['revenue_psc']:,.0f})")
        self.kpi_value_labels["ndg"].configure(
            text=f"{r['ndg_count']}/{self.params['job_batches']['J1']} (${r['revenue_ndg']:,.0f})")
        self.kpi_value_labels["busta"].configure(
            text=f"{r['busta_count']}/{self.params['job_batches']['J2']} (${r['revenue_busta']:,.0f})")
        self.kpi_value_labels["tard"].configure(
            text=f"J1={r['tardiness_min'].get('J1', 0)}m  J2={r['tardiness_min'].get('J2', 0)}m  (${r['tard_cost']:,.0f})")
        self.kpi_value_labels["stockout"].configure(
            text=f"L1={r['stockout_events'].get('L1', 0)}  L2={r['stockout_events'].get('L2', 0)}  (${r['stockout_cost']:,.0f})")
        self.kpi_value_labels["setup"].configure(
            text=f"{r.get('setup_events', 0)} events (${r.get('setup_cost', 0):,.0f})")
        self.kpi_value_labels["restock"].configure(
            text=f"{r.get('restock_count', 0)} completed"
                 f"{' | ACTIVE' if self.steppable.state.restock_busy > 0 else ''}")
        gc_parts = []
        for pair in sorted(self.steppable.state.gc_stock):
            stock = self.steppable.state.gc_stock[pair]
            cap = self.params.get("gc_capacity", {}).get(pair, "?")
            gc_parts.append(f"{pair[0]}_{pair[1]}={stock}/{cap}")
        self.kpi_value_labels["gc_stock"].configure(text="  ".join(gc_parts) if gc_parts else "n/a")
        self.kpi_value_labels["idle"].configure(text=f"{r['idle_min']} min (${r['idle_cost']:,.0f})")
        self.kpi_value_labels["over"].configure(text=f"{r['over_min']} min (${r['over_cost']:,.0f})")
        self.kpi_value_labels["batches"].configure(text=f"{len(self.steppable.state.completed_batches)} completed")
        self.kpi_value_labels["cancelled"].configure(text=str(len(self.steppable.state.cancelled_batches)))
        self.kpi_value_labels["ups"].configure(text=str(len(self.steppable.state.ups_events_fired)))
        self.kpi_value_labels["compute"].configure(
            text=f"{r['total_compute_ms']:,.0f} ms ({r['num_resolves']} resolves)")

    # ---- Pipeline ----

    def _refresh_pipeline_bar(self):
        state = self.steppable.state
        for lid in self.params["lines"]:
            mode = state.pipeline_mode.get(lid, "FREE")
            busy = state.pipeline_busy[lid]
            if mode == "RESTOCK" and busy > 0:
                rst = state.active_restock
                sku_txt = rst.sku if rst else "?"
                txt = f"Pipeline {lid}: RESTOCK {sku_txt} [{busy} left]"
                clr = "#7B1FA2"
            elif mode == "CONSUME" and busy > 0:
                batch = state.pipeline_batch[lid]
                if batch and hasattr(batch, "sku"):
                    txt = f"Pipeline {lid}: CONSUME {batch.sku} {batch.batch_id} [{busy} left]"
                else:
                    txt = f"Pipeline {lid}: BUSY [{busy} left]"
                clr = COLORS["warn"]
            elif busy > 0:
                txt = f"Pipeline {lid}: BUSY [{busy} left]"
                clr = COLORS["warn"]
            else:
                txt = f"Pipeline {lid}: FREE"
                clr = COLORS["good"]
            self.pipeline_text_var[lid].set(txt)
            self.pipeline_labels[lid].configure(fg=clr)

    # ---- MTO ----

    def _refresh_mto_panel(self):
        for jid in self.params["jobs"]:
            total = int(self.params["job_batches"][jid])
            done = self.steppable.completed_mto_counts[jid]
            rem = int(self.steppable.state.mto_remaining.get(jid, 0))
            last = self.steppable.mto_last_completion[jid]
            due = self.params["job_due"][jid]
            tard = self.steppable.kpi.tardiness_min.get(jid, 0.0)
            det = (f"{done}/{total} done | rem={rem} | due={due} | tard={tard}m | "
                   f"last: {last['roaster']} t={last['time'] if last['time'] is not None else '-'}")
            self.mto_widgets[jid]["progress"]["value"] = done
            self.mto_widgets[jid]["detail"].configure(
                text=det, fg=COLORS["bad"] if self.steppable.slot_index > due and rem > 0 else COLORS["muted"])

    # ---- Roaster detail popup ----

    def show_roaster_details(self, rid):
        state = self.steppable.state
        popup = tk.Toplevel(self.root)
        popup.title(f"{rid} details")
        popup.geometry("520x420")
        popup.configure(bg=COLORS["panel"])
        txt = scrolledtext.ScrolledText(popup, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True, padx=10, pady=10)
        batch = state.current_batch[rid]
        lines = [
            f"Roaster: {rid}", f"Status: {state.status[rid]}",
            f"Remaining: {state.remaining[rid]}", f"Current batch: {batch}",
            f"Last SKU: {state.last_sku[rid]}", f"Setup target: {state.setup_target_sku[rid]}",
            f"Needs decision: {state.needs_decision[rid]}", "", "Completed batches:",
        ]
        for b in state.completed_batches:
            if b.roaster == rid:
                lines.append(f"  - {b.sku} {b.batch_id} [{b.start},{b.end}) -> {b.output_line}")
        lines.append("")
        lines.append("Cancelled batches:")
        for b in state.cancelled_batches:
            if b.roaster == rid:
                lines.append(f"  - {b.sku} {b.batch_id} [{b.start},{b.end})")
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")

    # ---- UPS inject ----

    def open_inject_ups_dialog(self):
        if self.current_job:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Inject UPS")
        dlg.configure(bg=COLORS["panel"])
        dlg.transient(self.root)
        dlg.grab_set()
        dlg.resizable(False, False)
        rv = tk.StringVar(value=self.params["roasters"][0])
        dv = tk.IntVar(value=max(1, int(round(self.params.get("ups_mu", 15)))))
        ttk.Label(dlg, text="Roaster").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        ttk.Combobox(dlg, values=self.params["roasters"], textvariable=rv, state="readonly").grid(
            row=0, column=1, padx=10, pady=10)
        ttk.Label(dlg, text="Duration").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        tk.Spinbox(dlg, from_=1, to=60, textvariable=dv, width=8).grid(
            row=1, column=1, padx=10, pady=10, sticky="w")

        def inject():
            e = UPSEvent(t=int(self.steppable.slot_index), roaster_id=rv.get(), duration=int(dv.get()))
            self.steppable.add_ups_event(e)
            self.add_log_entry("ups", f"t={e.t}: [scheduled] UPS on {e.roaster_id} for {e.duration}m")
            dlg.destroy()

        ttk.Button(dlg, text="Inject", command=inject).grid(row=2, column=0, columnspan=2, pady=(0, 12))

    # ---- UPS preset ----

    def open_ups_preset_dialog(self):
        if self.current_job:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("UPS Preset")
        dlg.configure(bg=COLORS["panel"])
        dlg.geometry("420x340")
        dlg.transient(self.root)
        dlg.grab_set()
        lv = tk.DoubleVar(value=float(self.params.get("ups_lambda", 2.0)))
        mv = tk.DoubleVar(value=float(self.params.get("ups_mu", 20.0)))
        sv = tk.StringVar(value="42")
        ttk.Label(dlg, text="Lambda").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        tk.Spinbox(dlg, from_=0.0, to=20.0, increment=0.5, textvariable=lv, width=10).grid(
            row=0, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(dlg, text="Mean duration").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        tk.Spinbox(dlg, from_=1.0, to=60.0, increment=0.5, textvariable=mv, width=10).grid(
            row=1, column=1, padx=10, pady=10, sticky="w")
        ttk.Label(dlg, text="Seed").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        ttk.Entry(dlg, textvariable=sv, width=12).grid(row=2, column=1, padx=10, pady=10, sticky="w")
        lb = tk.Listbox(dlg, height=10, width=48)
        lb.grid(row=3, column=0, columnspan=3, padx=10, pady=8, sticky="nsew")

        def refresh_list(events):
            lb.delete(0, "end")
            for e in events:
                lb.insert("end", f"t={e.t:03d}  {e.roaster_id}  {e.duration}m")

        refresh_list(self.preset_ups_events)

        def generate():
            events = generate_ups_events(lv.get(), mv.get(), int(sv.get()))
            self.preset_ups_events = events
            refresh_list(events)
            if self.steppable.slot_index == 0:
                self.steppable.set_ups_events(list(events))
                self.refresh_all()

        def clear_all():
            self.preset_ups_events = []
            refresh_list([])
            if self.steppable.slot_index == 0:
                self.steppable.set_ups_events([])
                self.refresh_all()

        ttk.Button(dlg, text="Generate", command=generate).grid(row=4, column=0, padx=10, pady=10, sticky="w")
        ttk.Button(dlg, text="Clear All", command=clear_all).grid(row=4, column=1, padx=10, pady=10, sticky="w")
        ttk.Button(dlg, text="Close", command=dlg.destroy).grid(row=4, column=2, padx=10, pady=10, sticky="e")

    # ---- Export ----

    def export_current_run(self):
        if self.current_job:
            return
        od = filedialog.askdirectory(title="Choose export directory")
        if not od:
            return
        rid = f"dashboard_slot_{self.steppable.slot_index:03d}"
        paths = export_run(self.steppable.kpi.to_dict(), self.steppable.state,
                           self.params, self.steppable.ups_events, output_dir=od, run_id=rid)
        messagebox.showinfo("Export complete",
                            "Exported:\n" + "\n".join(f"- {Path(p).name}" for p in paths.values()))

    # ---- Compare ----

    def open_compare_dialog(self):
        if self.current_job:
            return
        self._start_bg("Running comparison...", self._bg_compare, self._show_compare)

    def _bg_compare(self):
        events = list(self.preset_ups_events)
        p = self.params
        t0 = time.perf_counter()
        dk, _ = SimulationEngine(p).run(DispatchingHeuristic(p), events)
        d_time = (time.perf_counter() - t0) * 1000

        q_time = 0.0
        qk = None
        try:
            from q_learning.q_strategy import QStrategy
            q_path = _PROJECT_ROOT / "q_learning" / "q_table.pkl"
            if q_path.exists():
                t0 = time.perf_counter()
                qk, _ = SimulationEngine(p).run(QStrategy(p, q_table_path=str(q_path)), events)
                q_time = (time.perf_counter() - t0) * 1000
        except Exception:
            pass

        return {
            "dispatching": dk.to_dict(), "d_time": d_time,
            "qlearning": qk.to_dict() if qk else None, "q_time": q_time,
            "ups_count": len(events),
        }

    def _show_compare(self, result):
        d = result["dispatching"]
        ql = result["qlearning"]
        dlg = tk.Toplevel(self.root)
        dlg.title("Strategy Comparison")
        dlg.geometry("620x420")
        dlg.configure(bg=COLORS["panel"])
        txt = scrolledtext.ScrolledText(dlg, wrap="word", font=("Consolas", 10))
        txt.pack(fill="both", expand=True, padx=12, pady=12)

        has_ql = ql is not None
        hdr = f"{'':18}{'Dispatching':>14}"
        if has_ql:
            hdr += f"{'Q-Learning':>14}"

        def _row(label, dk, qk=None, fmt=",.0f"):
            line = f"{label:18}{dk:>14{fmt}}"
            if qk is not None:
                line += f"{qk:>14{fmt}}"
            return line

        def _delta(base, val):
            diff = val - base
            return f"{diff:+,.0f}" if diff != 0 else "-"

        lines = [
            f"UPS preset events: {result['ups_count']}",
            "",
            hdr, "-" * len(hdr),
            _row("Net Profit $", d["net_profit"],
                 ql["net_profit"] if has_ql else None),
            _row("Revenue $", d["total_revenue"],
                 ql["total_revenue"] if has_ql else None),
            _row("Costs $", d["total_costs"],
                 ql["total_costs"] if has_ql else None),
            _row("PSC batches", d["psc_count"],
                 ql["psc_count"] if has_ql else None, "d"),
            _row("NDG batches", d["ndg_count"],
                 ql["ndg_count"] if has_ql else None, "d"),
            _row("BUSTA batches", d["busta_count"],
                 ql["busta_count"] if has_ql else None, "d"),
            _row("Setup events", d.get("setup_events", 0),
                 ql.get("setup_events", 0) if has_ql else None, "d"),
            _row("Setup cost $", d.get("setup_cost", 0),
                 ql.get("setup_cost", 0) if has_ql else None),
            _row("Restocks", d.get("restock_count", 0),
                 ql.get("restock_count", 0) if has_ql else None, "d"),
            _row("Tard cost $", d["tard_cost"],
                 ql["tard_cost"] if has_ql else None),
            _row("Stockout cost $", d["stockout_cost"],
                 ql["stockout_cost"] if has_ql else None),
            _row("Idle min", d["idle_min"],
                 ql["idle_min"] if has_ql else None),
            _row("Over min", d["over_min"],
                 ql["over_min"] if has_ql else None),
        ]

        comp_line = f"{'Compute (ms)':18}{result['d_time']:>14,.0f}"
        if has_ql:
            comp_line += f"{result['q_time']:>14,.0f}"
        lines.append(comp_line)

        if has_ql:
            lines.append("")
            delta_line = f"{'vs Dispatching':18}{'-':>14}"
            delta_line += f"{'$' + _delta(d['net_profit'], ql['net_profit']):>14}"
            lines.append(delta_line)

        txt.insert("1.0", "\n".join(lines))
        txt.configure(state="disabled")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch the reactive strategy dashboard.")
    parser.add_argument("--self-test", action="store_true", help="Create, step once, and exit.")
    args = parser.parse_args(argv)

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"Tkinter unavailable: {exc}") from exc

    if args.self_test:
        app = DashboardApp(root)
        root.update_idletasks()
        app.step_once()
        root.update()
        app.reset_simulation()
        root.update()
        root.destroy()
        return 0

    DashboardApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
