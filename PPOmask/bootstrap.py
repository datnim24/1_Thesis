"""Bootstrap script: install dependencies and verify imports."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent / "requirements.txt"

REQUIRED_IMPORTS = [
    ("gymnasium", "gymnasium"),
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("sb3_contrib", "sb3-contrib"),
    ("stable_baselines3", "stable-baselines3"),
    ("shimmy", "shimmy"),
    ("tensorboard", "tensorboard"),
    ("pytest", "pytest"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("plotly", "plotly"),
]


def install() -> None:
    print(f"[bootstrap] Installing from {REQUIREMENTS} ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print("[bootstrap] pip install completed.")


def verify() -> dict[str, str]:
    versions: dict[str, str] = {}
    missing: list[str] = []
    for module_name, pip_name in REQUIRED_IMPORTS:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "unknown")
            versions[pip_name] = ver
            print(f"  {pip_name:25s} {ver}")
        except ImportError:
            missing.append(pip_name)
            print(f"  {pip_name:25s} MISSING")
    if missing:
        raise ImportError(
            f"[bootstrap] Missing packages after install: {', '.join(missing)}. "
            "Run: pip install -r PPOmask/requirements.txt"
        )
    return versions


def main() -> None:
    install()
    print("\n[bootstrap] Verifying imports:")
    versions = verify()
    print(f"\n[bootstrap] All {len(versions)} packages OK.")


if __name__ == "__main__":
    main()
