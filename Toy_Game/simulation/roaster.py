"""
Roaster state machine for coffee roasting simulation.
States: IDLE, ROASTING, SETUP, DOWN
"""

from enum import Enum

# --- Constants ---
ROAST_TIME = 15        # P: batch processing time (minutes)
SETUP_TIME = 5         # σ: setup time between different SKUs
CONSUME_TIME = 3       # δ_con: pipeline consume duration
MAX_SHIFT_TIME = 480   # Total shift duration

# SKU types
SKU_PSC = "PSC"
SKU_NDG = "NDG"
SKU_BUSTA = "Busta"

# Roaster eligibility map
ROASTER_ELIGIBILITY = {
    "R1": [SKU_PSC, SKU_NDG],
    "R2": [SKU_PSC, SKU_NDG, SKU_BUSTA],
    "R3": [SKU_PSC],
    "R4": [SKU_PSC],
    "R5": [SKU_PSC],
}

# Line assignments
ROASTER_LINES = {
    "R1": "L1",
    "R2": "L1",
    "R3": "L2",
    "R4": "L2",
    "R5": "L2",
}

# Pipeline mapping (which pipeline a roaster uses for consume)
ROASTER_PIPELINE = {
    "R1": "L1",
    "R2": "L1",
    "R3": "L2",
    "R4": "L2",
    "R5": "L2",
}


class RoasterState(Enum):
    IDLE = "Idle"
    ROASTING = "Roasting"
    SETUP = "Setup"
    DOWN = "Down"


class Roaster:
    def __init__(self, roaster_id):
        self.id = roaster_id
        self.state = RoasterState.IDLE
        self.remaining_time = 0
        self.current_sku = None
        self.last_sku = SKU_PSC  # All roasters start as if last batch was PSC
        self.eligible_skus = ROASTER_ELIGIBILITY[roaster_id]
        self.line = ROASTER_LINES[roaster_id]
        self.pipeline = ROASTER_PIPELINE[roaster_id]
        self.pending_sku = None  # SKU queued after setup

        # Tracking
        self.batch_queue = []  # List of planned (sku, target_line) tuples
        self.batches_completed = 0
        self.total_roasting_time = 0
        self.total_setup_time = 0
        self.total_down_time = 0
        self.total_idle_time = 0

        # History for Gantt chart
        self.history = []  # List of {type, sku, start, end, target_line}

    def can_produce(self, sku):
        return sku in self.eligible_skus

    def needs_setup(self, sku):
        return self.last_sku is not None and self.last_sku != sku

    def start_batch(self, sku, current_time, target_line=None):
        """Start a batch. Returns True if started, False if setup needed first."""
        if self.state != RoasterState.IDLE:
            return False

        if not self.can_produce(sku):
            return False

        if self.needs_setup(sku):
            # Enter SETUP state first
            self.state = RoasterState.SETUP
            self.remaining_time = SETUP_TIME
            self.pending_sku = sku
            self.current_sku = None
            self.history.append({
                "type": "setup",
                "sku": f"{self.last_sku}→{sku}",
                "start": current_time,
                "end": current_time + SETUP_TIME,
                "target_line": None,
            })
            return True

        # No setup needed — start roasting immediately
        self.state = RoasterState.ROASTING
        self.remaining_time = ROAST_TIME
        self.current_sku = sku
        self.pending_sku = None

        # Determine output line
        if target_line is None:
            target_line = self.line
        
        self.history.append({
            "type": "roast",
            "sku": sku,
            "start": current_time,
            "end": current_time + ROAST_TIME,
            "target_line": target_line if sku == SKU_PSC else None,
        })
        return True

    def start_roasting_after_setup(self, current_time, target_line=None):
        """Called when setup completes to begin the actual roasting."""
        if self.pending_sku is None:
            return False

        sku = self.pending_sku
        self.state = RoasterState.ROASTING
        self.remaining_time = ROAST_TIME
        self.current_sku = sku
        self.last_sku = sku
        self.pending_sku = None

        if target_line is None:
            target_line = self.line

        self.history.append({
            "type": "roast",
            "sku": sku,
            "start": current_time,
            "end": current_time + ROAST_TIME,
            "target_line": target_line if sku == SKU_PSC else None,
        })
        return True

    def trigger_breakdown(self, duration, current_time):
        """Trigger a disruption/breakdown."""
        cancelled_sku = None

        if self.state == RoasterState.ROASTING:
            cancelled_sku = self.current_sku
            self.current_sku = None
            # Mark incomplete batch in history
            if self.history:
                self.history[-1]["cancelled"] = True

        elif self.state == RoasterState.SETUP:
            self.pending_sku = None

        self.state = RoasterState.DOWN
        self.remaining_time = duration
        self.current_sku = None

        self.history.append({
            "type": "down",
            "sku": None,
            "start": current_time,
            "end": current_time + duration,
            "target_line": None,
        })
        return cancelled_sku

    def tick(self, current_time):
        """Advance one time step. Returns events that occurred."""
        events = []

        if self.state == RoasterState.ROASTING:
            self.remaining_time -= 1
            self.total_roasting_time += 1
            if self.remaining_time <= 0:
                # Batch completes
                completed_sku = self.current_sku
                self.last_sku = self.current_sku
                self.current_sku = None
                self.state = RoasterState.IDLE
                self.remaining_time = 0
                self.batches_completed += 1
                # Find target line from history
                target_line = self.line
                if self.history:
                    last = self.history[-1]
                    if last.get("target_line"):
                        target_line = last["target_line"]
                events.append({
                    "type": "batch_complete",
                    "roaster": self.id,
                    "sku": completed_sku,
                    "time": current_time,
                    "target_line": target_line,
                })

        elif self.state == RoasterState.SETUP:
            self.remaining_time -= 1
            self.total_setup_time += 1
            if self.remaining_time <= 0:
                self.state = RoasterState.IDLE
                self.remaining_time = 0
                # Update last_sku to the new SKU (setup complete)
                if self.pending_sku:
                    self.last_sku = self.pending_sku
                events.append({
                    "type": "setup_complete",
                    "roaster": self.id,
                    "pending_sku": self.pending_sku,
                    "time": current_time,
                })

        elif self.state == RoasterState.DOWN:
            self.remaining_time -= 1
            self.total_down_time += 1
            if self.remaining_time <= 0:
                self.state = RoasterState.IDLE
                self.remaining_time = 0
                events.append({
                    "type": "repair_complete",
                    "roaster": self.id,
                    "time": current_time,
                })

        elif self.state == RoasterState.IDLE:
            self.total_idle_time += 1

        return events

    def get_utilization(self, current_time):
        """Compute roaster utilization percentage."""
        if current_time == 0:
            return 0.0
        return (self.total_roasting_time / current_time) * 100

    def to_dict(self, current_time=0):
        """Serialize state for JSON API."""
        return {
            "id": self.id,
            "state": self.state.value,
            "remaining_time": self.remaining_time,
            "current_sku": self.current_sku,
            "last_sku": self.last_sku,
            "pending_sku": self.pending_sku,
            "eligible_skus": self.eligible_skus,
            "line": self.line,
            "batches_completed": self.batches_completed,
            "utilization": round(self.get_utilization(current_time), 1),
            "batch_queue": self.batch_queue,
        }
