"""Tool-level masking for the RL-HH meta-agent.

Thin wrapper around ToolKit.compute_all() that returns both the boolean
mask and the cached action outputs (so tools are never executed twice per
decision point).
"""

from __future__ import annotations

from .tools import ToolKit


def compute_tool_mask(
    toolkit: ToolKit, state, roaster_id: str | None,
) -> tuple[list[int | None], list[bool]]:
    """Compute tool outputs and feasibility mask in a single pass.

    Returns
    -------
    tool_outputs : list[int | None]
        Action_id each tool would produce (None = infeasible).
    tool_mask : list[bool]
        True where the corresponding tool returned a valid action.
        Last entry (WAIT) is always True.
    """
    return toolkit.compute_all(state, roaster_id)
