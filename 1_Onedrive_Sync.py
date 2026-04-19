from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

SOURCE_DIR = Path(r"E:\CODE\Thesis\1_Thesis")
DEST_DIR = Path(r"C:\Users\Dat\OneDrive\1 Đại Học\1_THESIS\Code_Thesis_view")


def _remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _sync_with_shutil() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    for item in DEST_DIR.iterdir():
        if item.is_dir():
            shutil.rmtree(item, onerror=_remove_readonly)
        else:
            item.chmod(stat.S_IWRITE)
            item.unlink()

    shutil.copytree(SOURCE_DIR, DEST_DIR, dirs_exist_ok=True)


def _sync_with_robocopy() -> None:
    command = [
        "robocopy",
        str(SOURCE_DIR),
        str(DEST_DIR),
        "/MIR",
        "/R:2",
        "/W:2",
        "/XJ",
    ]

    completed = subprocess.run(command, check=False)
    if completed.returncode >= 8:
        raise RuntimeError(f"robocopy failed with exit code {completed.returncode}.")


def sync_onedrive() -> int:
    if not SOURCE_DIR.is_dir():
        print(f"Error: source directory does not exist: {SOURCE_DIR}")
        return 1

    DEST_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Syncing:\n  {SOURCE_DIR}\ninto:\n  {DEST_DIR}\n")

    try:
        _sync_with_robocopy()
        print("Sync completed with robocopy.")
        return 0
    except FileNotFoundError:
        print("robocopy is not available. Falling back to Python copy.")
        _sync_with_shutil()
        print("Sync completed with shutil.")
        return 0
    except Exception as exc:
        print(f"Sync failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(sync_onedrive())
