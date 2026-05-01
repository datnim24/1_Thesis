"""Pure CP-SAT solver (UPS-as-planned-downtime).

UPS events (if provided) are merged into the deterministic ``downtime_slots``
map, so the solver sees every outage as a pre-known planned stop. The solver's
objective value IS the schedule profit under env-equivalent accounting.
"""

from .runner import run_pure_cpsat

__all__ = ["run_pure_cpsat"]
