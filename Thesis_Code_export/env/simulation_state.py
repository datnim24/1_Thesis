"""Simulation state dataclasses for the GC-silo-extended environment."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BatchRecord:
    """One batch execution or planned queue item."""

    batch_id: tuple
    sku: str
    roaster: str
    start: int
    end: int
    output_line: Optional[str]
    is_mto: bool


@dataclass
class RestockRecord:
    """One GC restock operation."""

    line_id: str
    sku: str
    start: int
    end: int
    qty: int


@dataclass
class UPSEvent:
    """One UPS disruption event."""

    t: int
    roaster_id: str
    duration: int


@dataclass
class SimulationState:
    """Mutable state snapshot for the 480-slot simulation loop.

    Extends the original roaster+RC state with GC silo inventory,
    pipeline mode tracking, and global restock resource state.
    """

    t: int = 0

    # --- roaster state ---
    status: dict = field(default_factory=dict)
    remaining: dict = field(default_factory=dict)
    current_batch: dict = field(default_factory=dict)
    last_sku: dict = field(default_factory=dict)
    setup_target_sku: dict = field(default_factory=dict)

    # --- line pipeline state (now tracks mode) ---
    pipeline_busy: dict = field(default_factory=dict)
    pipeline_mode: dict = field(default_factory=dict)       # "FREE", "CONSUME", "RESTOCK"
    pipeline_batch: dict = field(default_factory=dict)       # batch or (line,sku) pair

    # --- downstream RC inventory ---
    rc_stock: dict = field(default_factory=dict)

    # --- upstream GC inventory ---
    gc_stock: dict = field(default_factory=dict)             # {(line, sku): int}

    # --- global restock resource ---
    restock_busy: int = 0
    active_restock: Optional[RestockRecord] = None

    # --- decision flags ---
    needs_decision: dict = field(default_factory=dict)
    needs_restock_decision: bool = False

    # --- schedule queues (for CP-SAT or external strategies) ---
    schedule_queue: dict[str, deque] = field(default_factory=dict)

    # --- MTO tracking ---
    mto_remaining: dict = field(default_factory=dict)
    mto_tardiness: dict = field(default_factory=dict)

    # --- history ---
    completed_batches: list = field(default_factory=list)
    cancelled_batches: list = field(default_factory=list)
    completed_restocks: list = field(default_factory=list)
    ups_events_fired: list = field(default_factory=list)
    trace: list = field(default_factory=list)
