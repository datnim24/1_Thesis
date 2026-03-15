"""
Disruption system for the roasting simulation.
Supports random (Poisson-arrival) and manual (user-triggered) disruptions.
"""

import random
import math


class DisruptionManager:
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.scheduled_events = []  # Pre-generated random events
        self.manual_events = []     # User-triggered events
        self.event_log = []         # All events that have occurred

    def generate_random_events(self, lam=3, min_duration=10, max_duration=30,
                                roaster_ids=None):
        """
        Generate random disruption events using Poisson process.
        lam: expected number of events per shift (λ)
        """
        if roaster_ids is None:
            roaster_ids = ["R1", "R2", "R3", "R4", "R5"]

        self.scheduled_events = []

        if lam <= 0:
            return

        # Mean inter-arrival time
        mean_interval = 480.0 / lam
        t_next = self.rng.expovariate(1.0 / mean_interval)

        while t_next < 480:
            roaster = self.rng.choice(roaster_ids)
            duration = self.rng.randint(min_duration, max_duration)
            self.scheduled_events.append({
                "time": int(t_next),
                "roaster": roaster,
                "duration": duration,
                "type": "random",
            })
            t_next += self.rng.expovariate(1.0 / mean_interval)

        # Sort by time
        self.scheduled_events.sort(key=lambda e: e["time"])

    def add_manual_disruption(self, roaster_id, duration, current_time):
        """User manually triggers a disruption."""
        event = {
            "time": current_time,
            "roaster": roaster_id,
            "duration": duration,
            "type": "manual",
        }
        self.manual_events.append(event)
        return event

    def get_events_at(self, current_time):
        """Get all disruption events scheduled for this time."""
        events = []

        # Check pre-generated random events
        for event in self.scheduled_events:
            if event["time"] == current_time:
                events.append(event)

        # Check manual events
        for event in self.manual_events:
            if event["time"] == current_time:
                events.append(event)

        return events

    def pop_manual_events(self):
        """Pop all pending manual events (they should be processed immediately)."""
        events = list(self.manual_events)
        self.manual_events = []
        return events

    def log_event(self, event, actual_time):
        """Record an event that was processed."""
        entry = dict(event)
        entry["processed_at"] = actual_time
        self.event_log.append(entry)

    def get_upcoming_events(self, current_time, window=60):
        """Preview upcoming random events (for UI display)."""
        return [
            e for e in self.scheduled_events
            if current_time <= e["time"] < current_time + window
        ]

    def get_event_log(self):
        """Get all processed events."""
        return list(self.event_log)

    def to_dict(self, current_time=0):
        """Serialize for JSON API."""
        return {
            "total_scheduled": len(self.scheduled_events),
            "total_occurred": len(self.event_log),
            "event_log": self.event_log[-10:],  # Last 10 events
            "upcoming": self.get_upcoming_events(current_time),
        }
