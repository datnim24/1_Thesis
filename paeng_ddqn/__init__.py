"""Paeng et al. (2021) Modified DDQN — primary key-reference comparison method
for the Nestlé Trị An batch roasting thesis.

Faithfully ported from `Paeng_DRL_Github/` (TF1.14) to PyTorch with the
problem-domain adaptations documented in `PORT_NOTES.md`.

Modules:
    agent     — ParameterSharingDQN, ReplayBuffer, PaengAgent (algorithm core)
    strategy  — build_paeng_state + PaengStrategy (engine adapter, state builder)
    train     — CLI training loop with seed rotation per episode
    evaluate  — single-seed report (Mode 1) + 100-seed aggregate (Mode 2)
"""
