"""Strategy wrapper that plugs RL-HH into SimulationEngine.run().

Provides decide(state, roaster_id) -> tuple  and decide_restock(state) -> tuple
so the canonical simulation engine can call the trained DDQN agent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from env.simulation_engine import SimulationEngine
from PPOmask.Engine.action_spec import ACTION_BY_ID, WAIT_ACTION_ID
from PPOmask.Engine.observation_spec import ObservationContext, build_observation

from .meta_agent import DuelingDDQNAgent
from .tools import ToolKit


class RLHHStrategy:
    """Plugs into SimulationEngine via decide(state, roaster) -> env_action tuple."""

    def __init__(
        self,
        agent: DuelingDDQNAgent,
        data,
        training: bool = False,
    ):
        self.agent = agent
        self.data = data
        self.params = data.to_env_params() if hasattr(data, "to_env_params") else data
        self._engine: SimulationEngine | None = None
        self._toolkit: ToolKit | None = None
        self.training = training

        # Tool selection stats (for evaluation diagnostics)
        self.tool_counts: dict[int, int] = {i: 0 for i in range(5)}

    def _get_engine(self) -> SimulationEngine:
        if self._engine is None:
            params = self.data.to_env_params() if hasattr(self.data, "to_env_params") else self.data
            self._engine = SimulationEngine(params)
        return self._engine

    def _get_toolkit(self) -> ToolKit:
        if self._toolkit is None:
            params = self.data.to_env_params() if hasattr(self.data, "to_env_params") else self.data
            self._toolkit = ToolKit(self._get_engine(), params)
        return self._toolkit

    # ------------------------------------------------------------------
    # SimulationEngine strategy interface
    # ------------------------------------------------------------------

    def decide(self, state, roaster_id: str) -> tuple:
        """Per-roaster decision: select tool via DDQN, return env_action tuple."""
        context = ObservationContext(kind="ROASTER", roaster_id=roaster_id)
        obs = build_observation(self.data, state, context)

        toolkit = self._get_toolkit()
        tool_outputs, tool_mask = toolkit.compute_all(state, roaster_id)

        tool_id = self.agent.select_tool(obs, tool_mask, training=self.training)
        self.tool_counts[tool_id] = self.tool_counts.get(tool_id, 0) + 1

        action_id = tool_outputs[tool_id]
        if action_id is None:
            action_id = WAIT_ACTION_ID
        return ACTION_BY_ID[action_id].env_action

    def decide_restock(self, state) -> tuple:
        """Global restock decision: select tool via DDQN, return env_action tuple."""
        context = ObservationContext(kind="RESTOCK")
        obs = build_observation(self.data, state, context)

        toolkit = self._get_toolkit()
        tool_outputs, tool_mask = toolkit.compute_all(state, None)

        tool_id = self.agent.select_tool(obs, tool_mask, training=self.training)
        self.tool_counts[tool_id] = self.tool_counts.get(tool_id, 0) + 1

        action_id = tool_outputs[tool_id]
        if action_id is None:
            return ("WAIT",)

        env_action = ACTION_BY_ID[action_id].env_action
        if env_action[0] == "START_RESTOCK":
            return env_action
        return ("WAIT",)
