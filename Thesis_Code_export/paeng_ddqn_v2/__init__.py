"""Paeng DDQN v2 — period-based Dueling DDQN port (Paeng 2021).

In active development. The agent uses Family-Based State (FBS) features and
parameter-sharing networks across roasters, with a period-based environment
wrapper (Paeng 2021 §3) instead of slot-by-slot dispatch.

Public modules:
- ``train``: training entry point (CLI: --name, --time, --seed).
- ``evaluate``: single-seed evaluation with universal-schema export.

Both modules currently raise ``NotImplementedError`` — they expose the
expected CLI surface so master_eval and scripts can call them once the
implementation lands.
"""
