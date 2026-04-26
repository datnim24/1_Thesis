"""Circular experience replay buffer for Dueling DDQN.

Stores transitions as contiguous numpy arrays for efficient batch sampling.
Each transition: (state, tool_id, reward, next_state, done, tool_mask, next_tool_mask).
"""

from __future__ import annotations

import numpy as np

from . import configs as C


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = C.BUFFER_SIZE,
        state_dim: int = C.INPUT_DIM,
        n_tools: int = C.N_TOOLS,
    ):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.tool_masks = np.zeros((capacity, n_tools), dtype=bool)
        self.next_tool_masks = np.zeros((capacity, n_tools), dtype=bool)
        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def store(
        self,
        state: np.ndarray,
        tool_id: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        tool_mask: list[bool],
        next_tool_mask: list[bool],
    ) -> None:
        idx = self._ptr
        self.states[idx] = state
        self.actions[idx] = tool_id
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.tool_masks[idx] = tool_mask
        self.next_tool_masks[idx] = next_tool_mask
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int = C.BATCH_SIZE,
    ) -> tuple[np.ndarray, ...]:
        """Uniform random batch.  Returns numpy arrays."""
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.tool_masks[indices],
            self.next_tool_masks[indices],
        )
