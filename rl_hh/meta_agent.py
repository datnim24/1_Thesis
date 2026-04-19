"""Dueling DDQN meta-agent for tool selection.

Handles epsilon-greedy exploration, masked Q-value selection, DDQN training
with soft target updates, and checkpoint save/load.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from . import configs as C
from .network import DuelingDDQN
from .replay_buffer import ReplayBuffer


class DuelingDDQNAgent:
    def __init__(
        self,
        lr: float = C.LR,
        gamma: float = C.GAMMA,
        buffer_size: int = C.BUFFER_SIZE,
        batch_size: int = C.BATCH_SIZE,
        epsilon_start: float = C.EPSILON_START,
        epsilon_end: float = C.EPSILON_END,
        epsilon_decay_frac: float = C.EPSILON_DECAY_FRAC,
        tau: float = C.TAU,
        grad_clip: float = C.GRAD_CLIP,
    ):
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.grad_clip = grad_clip

        self.online_net = DuelingDDQN()
        self.target_net = DuelingDDQN()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = Adam(self.online_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

        self.epsilon = epsilon_start
        self._eps_start = epsilon_start
        self._eps_end = epsilon_end
        self._eps_decay_frac = epsilon_decay_frac

        self.step_count = 0
        self._last_loss: float = 0.0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_tool(
        self,
        state_vec: np.ndarray,
        tool_mask: list[bool],
        training: bool = True,
    ) -> int:
        """Epsilon-greedy selection over masked Q-values."""
        valid_tools = [i for i, m in enumerate(tool_mask) if m]
        if not valid_tools:
            return C.N_TOOLS - 1  # WAIT fallback

        if training and random.random() < self.epsilon:
            return random.choice(valid_tools)

        with torch.no_grad():
            x = torch.as_tensor(state_vec, dtype=torch.float32).unsqueeze(0)
            q_values = self.online_net(x).squeeze(0)

        mask_tensor = torch.tensor(tool_mask, dtype=torch.bool)
        q_values[~mask_tensor] = -1e9
        return int(q_values.argmax().item())

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> float:
        """One gradient step using DDQN target with masked action selection."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        (states, actions, rewards, next_states,
         dones, masks, next_masks) = self.replay_buffer.sample(self.batch_size)

        states_t = torch.as_tensor(states)
        actions_t = torch.as_tensor(actions, dtype=torch.long)
        rewards_t = torch.as_tensor(rewards)
        next_states_t = torch.as_tensor(next_states)
        dones_t = torch.as_tensor(dones)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool)

        # Current Q-values for chosen tools
        q_current = self.online_net(states_t).gather(
            1, actions_t.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            # DDQN: online net selects, target net evaluates
            q_online_next = self.online_net(next_states_t)
            q_online_next[~next_masks_t] = -1e9
            best_tools = q_online_next.argmax(dim=1)

            q_target_next = self.target_net(next_states_t)
            q_next_val = q_target_next.gather(
                1, best_tools.unsqueeze(1),
            ).squeeze(1)

            targets = rewards_t + self.gamma * q_next_val * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_current, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.grad_clip)
        self.optimizer.step()

        # Soft update target network
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(self.tau * op.data + (1.0 - self.tau) * tp.data)

        self.step_count += 1
        self._last_loss = float(loss.item())
        return self._last_loss

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    def decay_epsilon(self, episode: int, total_episodes: int) -> None:
        """Linear decay from eps_start to eps_end over decay_frac of training."""
        decay_episodes = int(total_episodes * self._eps_decay_frac)
        if episode >= decay_episodes:
            self.epsilon = self._eps_end
        else:
            frac = episode / max(1, decay_episodes)
            self.epsilon = self._eps_start + frac * (self._eps_end - self._eps_start)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step_count": self.step_count,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt.get("epsilon", self._eps_end)
        self.step_count = ckpt.get("step_count", 0)
