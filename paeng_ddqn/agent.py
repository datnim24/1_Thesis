"""Paeng's Modified DDQN — algorithm core (network + replay + agent training).

Single-file port of three files from Paeng_DRL_Github/ to PyTorch:
    model/dqn.py::PDQN              -> ParameterSharingDQN
    agent/replay_buffer.py          -> ReplayBuffer
    agent/trainer.py::Trainer       -> PaengAgent
    config.py argparse defaults     -> PaengConfig dataclass

See PORT_NOTES.md for line-by-line provenance and the deltas vs. Paeng's
original code.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration dataclass — Paeng's config.py defaults with our overrides
# ---------------------------------------------------------------------------

@dataclass
class PaengConfig:
    """Hyperparameters for Paeng's Modified DDQN.

    Paeng's defaults (from his config.py) are kept verbatim where relevant.
    Overrides for our problem are flagged in the comments.
    """
    # Action / state dimensions ---------------------------------------------
    # State shape is (F=3 SKUs, F*2 + 32 = 50 cols per row) per Paeng
    # _getFamilyBasedState. Auxin holds last_action one-hot + last_reward + flag.
    state_F: int = 3                  # number of SKU families (PSC, NDG, BUSTA)
    state_W: int = 50                 # cols per family row (= 2*F^2 + 32 for F=3)
    auxin_dim: int = 10               # 8 (action one-hot) + 1 (last reward) + 1 (flag)
    action_dim: int = 8               # 3 SKUs + WAIT + 4 restock variants

    # Network architecture --------------------------------------------------
    hid_dims: tuple[int, ...] = (64, 32, 16)   # Cycle 13 (2026-04-28): reverted from (128,64,32) — bigger net collapsed in cycle 12
    is_double: bool = True            # Paeng's --is_double=True
    is_duel: bool = True              # Cycle 49 (2026-04-29): enable dueling head. The audit and run-seed sweep showed standard DDQN plateaus around +$45k; testing dueling on Paeng's parameter-sharing isolates the contribution.

    # Optimization ----------------------------------------------------------
    lr: float = 0.0025                # Cycle 8 (2026-04-28): reverted from 0.0010 (cycle 7 regressed)
    gamma: float = 0.99               # Cycle 43 (2026-04-29): reverted from 0.999 — too long horizon caused instability.
    batch_size: int = 32               # Cycle 29 (2026-04-29): reverted from 128 (cycle 28 didn't help)
    buffer_size: int = 10_000          # Cycle 29 (2026-04-29): reverted from 3k
    huber_delta: float = 0.5

    # Exploration / target sync --------------------------------------------
    eps_start: float = 0.2            # Paeng default (already low — no anneal from 1.0)
    eps_end: float = 0.05
    eps_ratio: float = 0.9            # anneal over 90% of total episodes
    warmup_timesteps: int = 24_000    # Paeng default
    freq_target_episodes: int = 50    # Paeng default — sync target every N episodes
    tau: float = 0.005                # OVERRIDE: soft τ-update (Paeng: hard τ=1.0)
    grad_clip: float = 10.0           # safety; Paeng has no explicit clip

    # Training schedule -----------------------------------------------------
    train_every_n_decisions: int = 1  # Paeng default: train each step post-warmup
    target_episode_estimate: int = 200_000  # used for ε decay denominator

    # Reward scaling --------------------------------------------------------
    # Paeng's reward (per-period tardiness / rnorm) lands in O(1) range. Our
    # raw profit-delta spans [-1500, +7000]; without scaling Q-values blow up.
    # `reward_scale` divides the per-decision profit-delta before storing.
    # Cycle 1 (2026-04-28): introduced reward_scale=1000 to fix the WAIT-collapse.
    reward_scale: float = 1000.0
    # Cycle 4 (2026-04-28): per-decision penalty subtracted from scaled reward when
    # WAIT (action 3) was chosen but a productive action (PSC=0/NDG=1/BUSTA=2) was
    # feasible. 0.05 in scaled units = $50 raw. ~1,300 decisions/episode → max ~$65k.
    idle_penalty: float = 0.05    # Cycle 43 (2026-04-29): re-enabled. Even with kpi_ref fix, the synthetic penalty helps break Q(WAIT) anchor at decision time (cycle 4's mechanism still applies).
    # Cycle 44 (2026-04-29): when False (default), per-decision reward in training uses
    # the revenue-only approximation from completed_batches (cycle 13's accidentally-effective
    # signal — sparse positive completions with no intra-episode cost noise). Function-approx
    # noise from full per-step Δprofit was destabilizing post-cycle-38 training. The terminal
    # transition still uses the true final net_profit so the agent learns total-cost behavior
    # via the long-horizon signal.
    use_kpi_for_reward: bool = False
    # Cycle 45 (2026-04-29): when True, decide_restock is delegated to DispatchingHeuristic
    # (matches q_learning_train.py pattern). When False, the agent learns restock too.
    # Cycle 55 (2026-04-29): reverted to False — cycle 54 dropped 100-seed mean from $182k to $97k
    # vs cycle 52 (agent-learned restock). The heuristic helped L1 supply but didn't compose
    # with dueling agent's roaster decisions on harder seeds.
    delegate_restock: bool = False
    # Cycle 16 reverted (2026-04-28): curriculum caused WAIT-only collapse; setting to 0 disables.
    curriculum_warmup_episodes: int = 0
    # Cycle 45 (2026-04-29): reverted to 0 since use_kpi_for_reward is now False — magnitudes
    # are back to cycle-13 range, no clipping needed.
    reward_clip: float = 0.0
    # Cycle 34: combo (lever A + lever E) regressed. Both reverted to 0 for cycle 36.
    force_productive_prob: float = 0.0
    productive_bonus: float = 0.0
    # Cycle 36 reverted (2026-04-29): stockout shaping fired too often and distorted policy.
    gc_low_threshold: float = 0.20
    stockout_alarm_penalty: float = 0.0
    # Cycle 37 (2026-04-29): switch optimizer RMSprop → Adam. Adam often finds flatter minima.
    use_adam: bool = True

    # Bookkeeping -----------------------------------------------------------
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Parameter-sharing Q-network (PyTorch port of model/dqn.py::PDQN)
# ---------------------------------------------------------------------------

class ParameterSharingDQN(nn.Module):
    """Per-row shared FC block, then fuse with auxin, then Q-head.

    Mirrors Paeng's `PDQN.base_encoder_cells` (lines 173-195) where the same
    hidden block is applied to each family row independently. PyTorch makes
    this trivial: pass through one nn.Sequential applied per slice along
    dim=1.
    """

    def __init__(self, config: PaengConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.state_W
        for h in config.hid_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.shared_block = nn.Sequential(*layers)
        self.encoded_dim = in_dim  # per-row encoded width (16 for default)

        # Fusion layer combines (F * encoded_dim) + auxin → action_dim Q-values
        fused_in = config.state_F * self.encoded_dim + config.auxin_dim
        if config.is_duel:
            self.value_head = nn.Sequential(
                nn.Linear(fused_in, 64), nn.ReLU(), nn.Linear(64, 1),
            )
            self.adv_head = nn.Sequential(
                nn.Linear(fused_in, 64), nn.ReLU(), nn.Linear(64, config.action_dim),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(fused_in, 64), nn.ReLU(), nn.Linear(64, config.action_dim),
            )

    def forward(self, state: torch.Tensor, auxin: torch.Tensor) -> torch.Tensor:
        """state: (B, F, W) ; auxin: (B, A) → Q: (B, action_dim)."""
        B, F_dim, W = state.shape
        # Apply shared block to each row, then concat along feature dim
        flat = state.reshape(B * F_dim, W)
        encoded = self.shared_block(flat)               # (B*F, encoded_dim)
        encoded = encoded.reshape(B, F_dim * self.encoded_dim)  # (B, F*encoded)
        fused = torch.cat([encoded, auxin], dim=1)       # (B, F*encoded + A)

        if self.config.is_duel:
            v = self.value_head(fused)                   # (B, 1)
            a = self.adv_head(fused)                     # (B, action_dim)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.q_head(fused)                        # (B, action_dim)


# ---------------------------------------------------------------------------
# Replay buffer (numpy circular; ports agent/replay_buffer.py:62-138)
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Numpy circular replay buffer.

    Stores 8-tuple per Paeng: (state, auxin, action, reward, terminal,
    next_state, next_auxin, next_feasibility_mask).
    """

    def __init__(self, capacity: int, config: PaengConfig):
        self.capacity = capacity
        self.cfg = config
        self.size = 0
        self._ptr = 0
        F_, W = config.state_F, config.state_W
        A = config.auxin_dim
        K = config.action_dim
        self.states = np.zeros((capacity, F_, W), dtype=np.float32)
        self.auxins = np.zeros((capacity, A), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.terminals = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, F_, W), dtype=np.float32)
        self.next_auxins = np.zeros((capacity, A), dtype=np.float32)
        self.next_masks = np.zeros((capacity, K), dtype=bool)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        auxin: np.ndarray,
        action: int,
        reward: float,
        terminal: bool,
        next_state: np.ndarray,
        next_auxin: np.ndarray,
        next_mask: np.ndarray,
    ) -> None:
        i = self._ptr
        self.states[i] = state
        self.auxins[i] = auxin
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.terminals[i] = float(bool(terminal))
        self.next_states[i] = next_state
        self.next_auxins[i] = next_auxin
        self.next_masks[i] = next_mask
        self._ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, n: int) -> dict[str, np.ndarray]:
        idx = np.random.randint(0, self.size, size=min(n, self.size))
        return {
            "states":      self.states[idx],
            "auxins":      self.auxins[idx],
            "actions":     self.actions[idx],
            "rewards":     self.rewards[idx],
            "terminals":   self.terminals[idx],
            "next_states": self.next_states[idx],
            "next_auxins": self.next_auxins[idx],
            "next_masks":  self.next_masks[idx],
        }


# ---------------------------------------------------------------------------
# Agent (ports agent/trainer.py::Trainer to a single self-contained class)
# ---------------------------------------------------------------------------

class PaengAgent:
    """Modified DDQN agent — wraps online + target nets, replay, optimizer.

    Public API mirrors what `paeng_ddqn.strategy.PaengStrategy` calls per
    decision plus what `paeng_ddqn.train.py` calls per episode. Single
    instance is reused across episodes (Paeng's pattern; replay persists).
    """

    def __init__(self, config: PaengConfig | None = None):
        self.cfg = config or PaengConfig()
        torch.manual_seed(0)

        self.online_net = ParameterSharingDQN(self.cfg).to(self.cfg.device)
        self.target_net = ParameterSharingDQN(self.cfg).to(self.cfg.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # Paeng uses RMSProp (`tf.train.RMSPropOptimizer`). Match here.
        if self.cfg.use_adam:
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.cfg.lr)
        else:
            self.optimizer = torch.optim.RMSprop(self.online_net.parameters(), lr=self.cfg.lr)

        self.replay = ReplayBuffer(self.cfg.buffer_size, self.cfg)
        self.timestep = 0
        self.epsilon = self.cfg.eps_start

    # ---- Action selection -----------------------------------------------

    def select_action(
        self,
        state: np.ndarray,         # (F, W)
        auxin: np.ndarray,         # (A,)
        feas_mask: np.ndarray,     # (action_dim,) bool
        training: bool,
    ) -> int:
        """ε-greedy with feasibility filter (Paeng trainer.py:102-156)."""
        feasible = np.where(feas_mask)[0]
        if len(feasible) == 0:
            # Engine-side fallback shouldn't happen if feasibility computed correctly,
            # but pick the WAIT action (id 3) as a safe noop.
            return 3
        if training and random.random() < self.epsilon:
            return int(random.choice(feasible.tolist()))
        # Greedy argmax over feasible
        with torch.no_grad():
            s_t = torch.from_numpy(state[None]).to(self.cfg.device)
            x_t = torch.from_numpy(auxin[None]).to(self.cfg.device)
            q = self.online_net(s_t, x_t).cpu().numpy()[0]
        q_masked = np.where(feas_mask, q, -1e9)
        return int(np.argmax(q_masked))

    # ---- Replay storage --------------------------------------------------

    def store_transition(self, *args, **kwargs) -> None:
        """Forwarded ReplayBuffer.add. See ReplayBuffer.add for signature."""
        self.replay.add(*args, **kwargs)
        self.timestep += 1

    # ---- Training step (Paeng trainer.py:291-330 train_network) ----------

    def train_step(self) -> float:
        """Returns Huber loss on a sampled minibatch (or 0.0 if pre-warmup)."""
        if self.timestep < self.cfg.warmup_timesteps:
            return 0.0
        if self.replay.size < self.cfg.batch_size:
            return 0.0
        if self.timestep % self.cfg.train_every_n_decisions != 0:
            return 0.0

        b = self.replay.sample_batch(self.cfg.batch_size)
        device = self.cfg.device

        states = torch.from_numpy(b["states"]).to(device)
        auxins = torch.from_numpy(b["auxins"]).to(device)
        actions = torch.from_numpy(b["actions"]).to(device)
        rewards = torch.from_numpy(b["rewards"]).to(device)
        terminals = torch.from_numpy(b["terminals"]).to(device)
        next_states = torch.from_numpy(b["next_states"]).to(device)
        next_auxins = torch.from_numpy(b["next_auxins"]).to(device)
        next_masks = torch.from_numpy(b["next_masks"]).to(device)

        # Q(s, a) for sampled actions
        q_pred = self.online_net(states, auxins).gather(
            1, actions.unsqueeze(1),
        ).squeeze(1)

        # Target Q(s', a') with Double-DQN action selection by online net
        with torch.no_grad():
            q_next_online = self.online_net(next_states, next_auxins)
            q_next_target = self.target_net(next_states, next_auxins)
            # Mask infeasible actions to -inf for argmax
            q_next_online_m = torch.where(
                next_masks, q_next_online, torch.full_like(q_next_online, -1e9),
            )
            if self.cfg.is_double:
                a_star = q_next_online_m.argmax(dim=1, keepdim=True)
                q_next = q_next_target.gather(1, a_star).squeeze(1)
            else:
                q_next_target_m = torch.where(
                    next_masks, q_next_target, torch.full_like(q_next_target, -1e9),
                )
                q_next = q_next_target_m.max(dim=1).values
            target = rewards + (1.0 - terminals) * self.cfg.gamma * q_next

        loss = F.smooth_l1_loss(q_pred, target, beta=self.cfg.huber_delta)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.cfg.grad_clip)
        self.optimizer.step()
        return float(loss.item())

    # ---- Target network sync (Paeng dqn.py:39-50, soft-update variant) ---

    def update_target_network(self) -> None:
        """Soft τ-update: target ← (1-τ)·target + τ·online (we override
        Paeng's hard τ=1.0 to be more stable on our shorter training runs).
        """
        tau = self.cfg.tau
        with torch.no_grad():
            for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * op.data)

    # ---- ε schedule (Paeng trainer.py:158-170 check_exploration) ---------

    def decay_epsilon(self, episode_idx: int, total_episodes: int | None = None) -> None:
        """Linear decay from eps_start → eps_end over eps_ratio·total_episodes."""
        n = total_episodes or self.cfg.target_episode_estimate
        decay_horizon = max(1, int(self.cfg.eps_ratio * n))
        if episode_idx >= decay_horizon:
            self.epsilon = self.cfg.eps_end
        else:
            frac = episode_idx / decay_horizon
            self.epsilon = self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # ---- Persistence -----------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.cfg.__dict__,
                "timestep": self.timestep,
                "epsilon": self.epsilon,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.timestep = int(ckpt.get("timestep", 0))
        self.epsilon = float(ckpt.get("epsilon", self.cfg.eps_end))

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "PaengAgent":
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        # Restore config; tuples are stored as lists in the dict
        if "hid_dims" in cfg_dict and isinstance(cfg_dict["hid_dims"], list):
            cfg_dict["hid_dims"] = tuple(cfg_dict["hid_dims"])
        cfg = PaengConfig(**cfg_dict)
        agent = cls(cfg)
        agent.online_net.load_state_dict(ckpt["online_state_dict"])
        agent.target_net.load_state_dict(ckpt["target_state_dict"])
        agent.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        agent.timestep = int(ckpt.get("timestep", 0))
        agent.epsilon = float(ckpt.get("epsilon", cfg.eps_end))
        return agent
