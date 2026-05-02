"""Paeng DDQN v2 — period-based parameter-sharing Dueling DDQN (Paeng 2021 port).

Period-based decisions (every 11-minute period vs slot-by-slot), Family-Based
State (FBS) features, parameter-sharing dueling network across roasters,
reward shaping per Paeng 2021 §3.

Public modules:
- ``train``: full training pipeline (config + agent + strategy + training loop).
- ``evaluate``: single-seed schema-export + multi-seed aggregate evaluation.
"""
