"""
RC Silo inventory tracking for each production line.
Handles stock balance: initial + completed_batches - consumption_events
"""

import math


class RCSilo:
    def __init__(self, line_id, initial_stock=12, max_buffer=40, consumption_rate=5.1):
        self.line_id = line_id
        self.stock = initial_stock
        self.initial_stock = initial_stock
        self.max_buffer = max_buffer
        self.consumption_rate = consumption_rate  # ρ: minutes per batch consumed

        # Pre-compute consumption schedule
        self.consumption_events = self._compute_consumption_schedule()

        # Tracking
        self.stock_history = []  # [(time, stock_level)]
        self.stockout_count = 0
        self.stockout_duration = 0
        self.total_produced = 0
        self.total_consumed = 0

    def _compute_consumption_schedule(self):
        """Pre-compute E_l = {floor(i * ρ_l) | i = 1, 2, ...}"""
        events = set()
        max_events = int(480 / self.consumption_rate)
        for i in range(1, max_events + 1):
            t = math.floor(i * self.consumption_rate)
            if t < 480:
                events.add(t)
        return events

    def update_consumption_schedule(self, new_rate):
        """Recompute consumption schedule with new rate."""
        self.consumption_rate = new_rate
        self.consumption_events = self._compute_consumption_schedule()

    def add_batch(self, current_time):
        """A PSC batch completed — add 1 to RC stock."""
        if self.stock < self.max_buffer:
            self.stock += 1
            self.total_produced += 1
            return True
        return False  # Overflow — cannot add

    def process_consumption(self, current_time):
        """Check if consumption event occurs at this time. Returns True if consumed."""
        if current_time in self.consumption_events:
            self.stock -= 1
            self.total_consumed += 1

            if self.stock < 0:
                self.stockout_count += 1
                self.stockout_duration += 1
            return True
        
        # Track ongoing stockout duration (even without new consumption event)
        if self.stock < 0:
            self.stockout_duration += 1

        return False

    def record_history(self, current_time):
        """Record current stock level for charting."""
        self.stock_history.append((current_time, self.stock))

    def would_overflow(self, additional=1):
        """Check if adding `additional` batches would exceed buffer."""
        return (self.stock + additional) > self.max_buffer

    def get_status(self):
        """Get stock status color."""
        ratio = self.stock / self.max_buffer if self.max_buffer > 0 else 0
        if self.stock <= 0:
            return "stockout"
        elif ratio < 0.25:
            return "low"
        elif ratio < 0.5:
            return "warning"
        else:
            return "healthy"

    def to_dict(self):
        """Serialize for JSON API."""
        return {
            "line_id": self.line_id,
            "stock": self.stock,
            "max_buffer": self.max_buffer,
            "consumption_rate": round(self.consumption_rate, 2),
            "status": self.get_status(),
            "fill_percent": round(max(0, self.stock / self.max_buffer * 100), 1),
            "stockout_count": self.stockout_count,
            "stockout_duration": self.stockout_duration,
            "total_produced": self.total_produced,
            "total_consumed": self.total_consumed,
        }
