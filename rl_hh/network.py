"""Dueling DDQN network with grouped input architecture (Ren & Liu 2024).

Input:  33 continuous features (same observation as PPO).
Output: 5 Q-values (one per tool).

Feature groups:
  A — Roaster block  : obs[1:16]  (15 features)
  B — Inventory block : obs[16:27] (11 features)
  C — Context block   : obs[0:1] + obs[27:33] (7 features)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import configs as C


class DuelingDDQN(nn.Module):
    def __init__(
        self,
        n_tools: int = C.N_TOOLS,
        roaster_dim: int = C.ROASTER_DIM,
        inventory_dim: int = C.INVENTORY_DIM,
        context_dim: int = C.CONTEXT_DIM,
    ):
        super().__init__()

        # Group A — Roaster block
        self.roaster_net = nn.Sequential(
            nn.Linear(roaster_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Group B — Inventory block
        self.inventory_net = nn.Sequential(
            nn.Linear(inventory_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Group C — Context + time block
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
        )

        # Merge: cat(32, 32, 16) = 80
        self.merge = nn.Sequential(
            nn.Linear(80, 128),
            nn.ReLU(),
        )

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_tools),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.  x: (batch, 33) -> Q: (batch, n_tools)."""
        roaster_in = x[:, C.ROASTER_FEATURES]          # (B, 15)
        inventory_in = x[:, C.INVENTORY_FEATURES]       # (B, 11)
        context_in = x[:, C.CONTEXT_FEATURES_IDX]       # (B, 7)

        r = self.roaster_net(roaster_in)
        i = self.inventory_net(inventory_in)
        c = self.context_net(context_in)

        merged = self.merge(torch.cat([r, i, c], dim=1))

        v = self.value_stream(merged)          # (B, 1)
        a = self.advantage_stream(merged)      # (B, n_tools)

        # Q = V + (A - mean(A))
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
