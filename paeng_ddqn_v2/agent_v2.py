"""Paeng DDQN agent — faithful to 2021 paper.

Double DQN + Dueling architecture per Algorithm 1.
Period-based decisions with hard target-network synchronization (τ=1.0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PaengConfigV2:
    """Hyperparameters faithful to Paeng et al. (2021) Table II."""

    action_dim: int = 9
    state_rows: int = 3
    state_cols: int = 35   # 25 paper-faithful + 10 domain features (RC/GC/pipeline/time/UPS)
    sf_dim: int = 3

    lr: float = 0.0025
    gamma: float = 1.0
    batch_size: int = 64
    buffer_size: int = 100_000
    warmup_timesteps: int = 2_000
    freq_target_episodes: int = 50

    eps_start: float = 0.2
    eps_end: float = 0.0
    eps_ratio: float = 0.8

    is_double: bool = True
    is_duel: bool = True
    tau: float = 1.0
    huber_delta: float = 1.0

    target_episode_estimate: int = 100_000
    period_length: int = 11


class ParameterSharingDQNV2(nn.Module):
    """Parameter-sharing DQN per Paeng paper.

    Input: state (B, 3, 25) + vec(Sf) auxin (B, 3).
    Shared encoder per family row, then fusion and dueling heads.
    """

    def __init__(self, state_rows: int, state_cols: int, sf_dim: int, action_dim: int, is_duel: bool = True):
        super().__init__()
        self.state_rows = state_rows
        self.state_cols = state_cols
        self.sf_dim = sf_dim
        self.action_dim = action_dim
        self.is_duel = is_duel

        # Shared encoder: applies to each row independently
        self.encoder = nn.Sequential(
            nn.Linear(state_cols, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        # Fusion layer after row concatenation + sf auxin
        fusion_input_dim = state_rows * 16 + sf_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 32),
            nn.ReLU(),
        )

        if is_duel:
            # Dueling heads
            self.v_head = nn.Linear(32, 1)
            self.a_head = nn.Linear(32, action_dim)
        else:
            # Standard Q-head
            self.q_head = nn.Linear(32, action_dim)

    def forward(self, state: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, 3, 25)
            sf: (B, 3)
        Returns:
            q_values: (B, 9)
        """
        B = state.size(0)

        # Encode each row
        encoded_rows = []
        for i in range(self.state_rows):
            row_i = state[:, i, :]  # (B, 25)
            encoded_i = self.encoder(row_i)  # (B, 16)
            encoded_rows.append(encoded_i)

        # Concatenate rows + sf auxin
        encoded_concat = torch.cat(encoded_rows + [sf], dim=1)  # (B, 48+3) = (B, 51)
        fused = self.fusion(encoded_concat)  # (B, 32)

        if self.is_duel:
            v = self.v_head(fused)  # (B, 1)
            a = self.a_head(fused)  # (B, 9)
            q = v + (a - a.mean(dim=1, keepdim=True))
            return q
        else:
            return self.q_head(fused)


class ReplayBufferV2:
    """Circular replay buffer for experience tuples."""

    def __init__(self, buffer_size: int, state_shape: tuple, sf_shape: tuple, action_dim: int):
        self.buffer_size = buffer_size
        self.state_shape = state_shape
        self.sf_shape = sf_shape
        self.action_dim = action_dim
        self.idx = 0
        self.size = 0

        # Pre-allocate numpy arrays
        self.state = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.sf = np.zeros((buffer_size, *sf_shape), dtype=np.float32)
        self.action = np.zeros((buffer_size,), dtype=np.int64)
        self.reward = np.zeros((buffer_size,), dtype=np.float32)
        self.terminal = np.zeros((buffer_size,), dtype=bool)
        self.next_state = np.zeros((buffer_size, *state_shape), dtype=np.float32)
        self.next_sf = np.zeros((buffer_size, *sf_shape), dtype=np.float32)
        self.next_feas = np.zeros((buffer_size, action_dim), dtype=bool)

    def add(self, state, sf, action, reward, terminal, next_state, next_sf, next_feas):
        self.state[self.idx] = state
        self.sf[self.idx] = sf
        self.action[self.idx] = action
        self.reward[self.idx] = reward
        self.terminal[self.idx] = terminal
        self.next_state[self.idx] = next_state
        self.next_sf[self.idx] = next_sf
        self.next_feas[self.idx] = next_feas

        self.idx = (self.idx + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample_batch(self, batch_size: int):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.from_numpy(self.state[indices]),
            torch.from_numpy(self.sf[indices]),
            torch.from_numpy(self.action[indices]),
            torch.from_numpy(self.reward[indices]),
            torch.from_numpy(self.terminal[indices]),
            torch.from_numpy(self.next_state[indices]),
            torch.from_numpy(self.next_sf[indices]),
            torch.from_numpy(self.next_feas[indices]),
        )

    def __len__(self):
        return self.size


class PaengAgentV2:
    """Paeng DDQN agent with period-based decisions."""

    def __init__(self, config: PaengConfigV2, device: str = "cpu"):
        self.config = config
        self.device = device

        # Networks
        self.online = ParameterSharingDQNV2(
            config.state_rows, config.state_cols, config.sf_dim,
            config.action_dim, is_duel=config.is_duel
        ).to(device)
        self.target = ParameterSharingDQNV2(
            config.state_rows, config.state_cols, config.sf_dim,
            config.action_dim, is_duel=config.is_duel
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        # Optimizer (paper doesn't specify, use Adam as standard)
        self.optimizer = optim.Adam(self.online.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss(beta=config.huber_delta)

        # Replay buffer
        self.replay = ReplayBufferV2(
            config.buffer_size,
            state_shape=(config.state_rows, config.state_cols),
            sf_shape=(config.sf_dim,),
            action_dim=config.action_dim,
        )

        # Training state
        self.epsilon = config.eps_start
        self.timestep = 0
        self.episode = 0

    def select_action(
        self,
        state: np.ndarray,
        sf: np.ndarray,
        feas_mask: np.ndarray,
        training: bool = True,
    ) -> int:
        """ε-greedy action selection with feasibility mask.

        Args:
            state: (3, 25)
            sf: (3,)
            feas_mask: (9,) boolean array
            training: if True, apply ε-greedy; else greedy

        Returns:
            action_id: int in [0, 8]
        """
        if training and np.random.random() < self.epsilon:
            # Random from feasible
            feasible_actions = np.where(feas_mask)[0]
            if len(feasible_actions) == 0:
                feasible_actions = np.arange(self.config.action_dim)
            return int(np.random.choice(feasible_actions))
        else:
            # Greedy: argmax over feasible
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1, 3, 25)
                sf_t = torch.from_numpy(sf).unsqueeze(0).to(self.device)  # (1, 3)
                q = self.online(state_t, sf_t)  # (1, 9)
                q_np = q.cpu().numpy()[0]  # (9,)

            # Mask infeasible actions with -inf
            q_masked = q_np.copy()
            q_masked[~feas_mask] = -np.inf

            # Argmax over masked Q-values
            action = int(np.argmax(q_masked))
            return action

    def store_transition(
        self,
        state,
        sf,
        action,
        reward,
        terminal,
        next_state,
        next_sf,
        next_feas,
    ):
        """Store transition in replay buffer."""
        self.replay.add(state, sf, action, reward, terminal, next_state, next_sf, next_feas)
        self.timestep += 1

    def train_step(self) -> float:
        """One SGD step with Double-Q target.

        Returns:
            loss value (float)
        """
        if len(self.replay) < self.config.batch_size or self.timestep < self.config.warmup_timesteps:
            return 0.0

        # Sample batch
        state, sf, action, reward, terminal, next_state, next_sf, next_feas = self.replay.sample_batch(
            self.config.batch_size
        )

        state = state.to(self.device)
        sf = sf.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        terminal = terminal.to(self.device)
        next_state = next_state.to(self.device)
        next_sf = next_sf.to(self.device)
        next_feas = next_feas.to(self.device)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            next_q_online = self.online(next_state, next_sf)  # (B, 9)

            # Mask infeasible actions
            next_q_online_masked = next_q_online.clone()
            next_q_online_masked[~next_feas] = -1e9

            next_action = torch.argmax(next_q_online_masked, dim=1)  # (B,)

            next_q_target = self.target(next_state, next_sf)  # (B, 9)
            next_q_max = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1)  # (B,)

            # Compute target
            target_q = reward + self.config.gamma * next_q_max * (~terminal).float()  # (B,)

        # Compute online Q and loss
        q_online = self.online(state, sf)  # (B, 9)
        q_selected = q_online.gather(1, action.unsqueeze(1)).squeeze(1)  # (B,)

        loss = self.loss_fn(q_selected, target_q)

        # SGD
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Hard copy: target := online (τ=1.0)."""
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self, episode_idx: int, max_episodes: int):
        """Linear decay from eps_start to eps_end over eps_ratio fraction of episodes."""
        decay_episodes = int(self.config.eps_ratio * max_episodes)
        if episode_idx < decay_episodes:
            self.epsilon = self.config.eps_start - (
                self.config.eps_start - self.config.eps_end
            ) * (episode_idx / decay_episodes)
        else:
            self.epsilon = self.config.eps_end

    def save_checkpoint(self, path: Path | str):
        """Save agent state."""
        path = Path(path)
        torch.save(
            {
                "online_state_dict": self.online.state_dict(),
                "target_state_dict": self.target.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "epsilon": self.epsilon,
                "timestep": self.timestep,
                "episode": self.episode,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str):
        """Load agent state."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online_state_dict"])
        self.target.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.epsilon = ckpt.get("epsilon", self.config.eps_start)
        self.timestep = ckpt.get("timestep", 0)
        self.episode = ckpt.get("episode", 0)

    @classmethod
    def from_checkpoint(cls, path: Path | str, device: str = "cpu") -> PaengAgentV2:
        """Load agent from checkpoint."""
        path = Path(path)
        ckpt = torch.load(path, map_location=device)
        config_dict = ckpt["config"]
        config = PaengConfigV2(**config_dict)
        agent = cls(config, device=device)
        agent.load_checkpoint(path)
        return agent
