"""Numpy-only forward pass for the Dueling DDQN network.

Extracts weights from the PyTorch model and runs inference with pure numpy.
Eliminates all PyTorch call overhead (~0.2ms per call) which dominates
for a tiny 33k-param network.

Usage:
    np_net = NumpyDuelingDDQN(agent.online_net)
    q_values = np_net.forward(obs_batch)  # (N, 5) ndarray
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn


def _extract_linear(mod: nn.Linear) -> tuple[np.ndarray, np.ndarray]:
    """Extract weight and bias from a Linear layer as numpy arrays."""
    w = mod.weight.detach().cpu().numpy().copy()
    b = mod.bias.detach().cpu().numpy().copy()
    return w, b


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, out=x)


class NumpyDuelingDDQN:
    """Pure-numpy forward pass for the Dueling DDQN."""

    __slots__ = (
        "r_w1", "r_b1", "r_w2", "r_b2",
        "i_w1", "i_b1", "i_w2", "i_b2",
        "c_w1", "c_b1",
        "m_w1", "m_b1",
        "v_w1", "v_b1", "v_w2", "v_b2",
        "a_w1", "a_b1", "a_w2", "a_b2",
        "_ctx_idx", "_bufs",
    )

    def __init__(self, torch_net):
        # Group A — Roaster
        self.r_w1, self.r_b1 = _extract_linear(torch_net.roaster_net[0])
        self.r_w2, self.r_b2 = _extract_linear(torch_net.roaster_net[2])
        # Group B — Inventory
        self.i_w1, self.i_b1 = _extract_linear(torch_net.inventory_net[0])
        self.i_w2, self.i_b2 = _extract_linear(torch_net.inventory_net[2])
        # Group C — Context
        self.c_w1, self.c_b1 = _extract_linear(torch_net.context_net[0])
        # Merge
        self.m_w1, self.m_b1 = _extract_linear(torch_net.merge[0])
        # Value stream
        self.v_w1, self.v_b1 = _extract_linear(torch_net.value_stream[0])
        self.v_w2, self.v_b2 = _extract_linear(torch_net.value_stream[2])
        # Advantage stream
        self.a_w1, self.a_b1 = _extract_linear(torch_net.advantage_stream[0])
        self.a_w2, self.a_b2 = _extract_linear(torch_net.advantage_stream[2])

        # Context feature indices: [0, 27, 28, 29, 30, 31, 32]
        from . import configs as C
        self._ctx_idx = np.array(C.CONTEXT_FEATURES_IDX, dtype=np.intp)

        # Pre-allocated buffers for forward_single (zero-alloc inference)
        self._bufs = {
            "r1": np.empty(64, dtype=np.float32),
            "r2": np.empty(32, dtype=np.float32),
            "i1": np.empty(64, dtype=np.float32),
            "i2": np.empty(32, dtype=np.float32),
            "c1": np.empty(16, dtype=np.float32),
            "cat": np.empty(80, dtype=np.float32),
            "m1": np.empty(128, dtype=np.float32),
            "vh": np.empty(64, dtype=np.float32),
            "v": np.empty(1, dtype=np.float32),
            "ah": np.empty(64, dtype=np.float32),
            "a": np.empty(5, dtype=np.float32),
            "q": np.empty(5, dtype=np.float32),
        }

    def sync(self, torch_net) -> None:
        """Re-extract weights after training step."""
        self.r_w1, self.r_b1 = _extract_linear(torch_net.roaster_net[0])
        self.r_w2, self.r_b2 = _extract_linear(torch_net.roaster_net[2])
        self.i_w1, self.i_b1 = _extract_linear(torch_net.inventory_net[0])
        self.i_w2, self.i_b2 = _extract_linear(torch_net.inventory_net[2])
        self.c_w1, self.c_b1 = _extract_linear(torch_net.context_net[0])
        self.m_w1, self.m_b1 = _extract_linear(torch_net.merge[0])
        self.v_w1, self.v_b1 = _extract_linear(torch_net.value_stream[0])
        self.v_w2, self.v_b2 = _extract_linear(torch_net.value_stream[2])
        self.a_w1, self.a_b1 = _extract_linear(torch_net.advantage_stream[0])
        self.a_w2, self.a_b2 = _extract_linear(torch_net.advantage_stream[2])

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, 33) -> Q: (N, 5).  All float32."""
        # Group A: x[:, 1:16] -> (N, 32)
        r = x[:, 1:16] @ self.r_w1.T + self.r_b1
        _relu(r)
        r = r @ self.r_w2.T + self.r_b2
        _relu(r)

        # Group B: x[:, 16:27] -> (N, 32)
        inv = x[:, 16:27] @ self.i_w1.T + self.i_b1
        _relu(inv)
        inv = inv @ self.i_w2.T + self.i_b2
        _relu(inv)

        # Group C: x[:, ctx_idx] -> (N, 16)
        ctx = x[:, self._ctx_idx] @ self.c_w1.T + self.c_b1
        _relu(ctx)

        # Merge
        merged = np.concatenate([r, inv, ctx], axis=1) @ self.m_w1.T + self.m_b1
        _relu(merged)

        # Value
        v = merged @ self.v_w1.T + self.v_b1
        _relu(v)
        v = v @ self.v_w2.T + self.v_b2  # (N, 1)

        # Advantage
        a = merged @ self.a_w1.T + self.a_b1
        _relu(a)
        a = a @ self.a_w2.T + self.a_b2  # (N, 5)

        # Q = V + A - mean(A)
        return v + a - a.mean(axis=1, keepdims=True)

    def forward_single(self, x: np.ndarray) -> np.ndarray:
        """x: (33,) -> Q: (5,).  Pre-allocated buffers for zero-alloc inference."""
        b = self._bufs
        # Group A
        np.dot(self.r_w1, x[1:16], out=b["r1"])
        b["r1"] += self.r_b1
        np.maximum(b["r1"], 0.0, out=b["r1"])
        np.dot(self.r_w2, b["r1"], out=b["r2"])
        b["r2"] += self.r_b2
        np.maximum(b["r2"], 0.0, out=b["r2"])

        # Group B
        np.dot(self.i_w1, x[16:27], out=b["i1"])
        b["i1"] += self.i_b1
        np.maximum(b["i1"], 0.0, out=b["i1"])
        np.dot(self.i_w2, b["i1"], out=b["i2"])
        b["i2"] += self.i_b2
        np.maximum(b["i2"], 0.0, out=b["i2"])

        # Group C
        np.dot(self.c_w1, x[self._ctx_idx], out=b["c1"])
        b["c1"] += self.c_b1
        np.maximum(b["c1"], 0.0, out=b["c1"])

        # Merge: cat(32, 32, 16) = 80 -> 128
        cat = b["cat"]
        cat[:32] = b["r2"]
        cat[32:64] = b["i2"]
        cat[64:] = b["c1"]
        np.dot(self.m_w1, cat, out=b["m1"])
        b["m1"] += self.m_b1
        np.maximum(b["m1"], 0.0, out=b["m1"])

        # V stream
        np.dot(self.v_w1, b["m1"], out=b["vh"])
        b["vh"] += self.v_b1
        np.maximum(b["vh"], 0.0, out=b["vh"])
        np.dot(self.v_w2, b["vh"], out=b["v"])
        b["v"] += self.v_b2

        # A stream
        np.dot(self.a_w1, b["m1"], out=b["ah"])
        b["ah"] += self.a_b1
        np.maximum(b["ah"], 0.0, out=b["ah"])
        np.dot(self.a_w2, b["ah"], out=b["a"])
        b["a"] += self.a_b2

        # Q = V + A - mean(A)
        q = b["q"]
        q[:] = b["v"][0] + b["a"] - b["a"].mean()
        return q
