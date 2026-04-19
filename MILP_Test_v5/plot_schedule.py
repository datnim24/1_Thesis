"""Plot 1-3 MILP_Test_v5 result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROASTER_ORDER = ["R1", "R2", "R3", "R4", "R5"]
LINE_ORDER = ["L1", "L2"]
SKU_COLORS = {
    "PSC": "#4C78A8",
    "NDG": "#F58518",
    "BUSTA": "#54A24B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 1 to 3 MILP_Test_v5 result JSON files.")
    parser.add_argument("json_files", nargs="+", help="One to three result JSON files.")
    parser.add_argument("--save", default=None, metavar="PATH", help="Optional image output path.")
    parser.add_argument("--no-show", action="store_true", help="Save only; do not open a window.")
    args = parser.parse_args()
    if not 1 <= len(args.json_files) <= 3:
        parser.error("Please provide between 1 and 3 JSON files.")
    return args


def load_result(path_str: str) -> dict:
    path = Path(path_str)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_path"] = path
    return data


def plot_results(
    results: list[dict[str, Any]],
    save_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    normalized: list[dict[str, Any]] = []
    for idx, result in enumerate(results, start=1):
        current = dict(result)
        if "_path" not in current:
            current["_path"] = Path(f"result_{idx}.json")
        normalized.append(current)

    if save_path or not show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        nrows=len(normalized),
        ncols=1,
        figsize=(15, 4.2 * len(normalized)),
        sharex=True,
    )
    if len(normalized) == 1:
        axes = [axes]

    for ax, result in zip(axes, normalized):
        build_subplot(ax, result)

    add_legend(fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path: Path | None = None
    if save_path:
        output_path = Path(save_path)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        print(f"Saved plot to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def build_subplot(ax, result: dict) -> None:
    schedule = result.get("schedule", [])
    restocks = result.get("restocks", [])
    roasters = [roaster for roaster in ROASTER_ORDER if any(entry["roaster"] == roaster for entry in schedule)]
    if not roasters:
        roasters = ROASTER_ORDER
    y_positions = {roaster: idx for idx, roaster in enumerate(roasters)}
    restock_y = len(roasters)

    for entry in schedule:
        start = int(entry["start"])
        end = int(entry["end"])
        width = end - start
        color = SKU_COLORS.get(entry["sku"], "#9C755F")
        hatch = "//" if entry.get("is_mto") else None
        ax.barh(
            y_positions[entry["roaster"]],
            width,
            left=start,
            height=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch,
        )

    for rst in restocks:
        color = "#E45756" if rst["line_id"] == "L1" else "#B279A2"
        ax.barh(
            restock_y,
            int(rst["end"]) - int(rst["start"]),
            left=int(rst["start"]),
            height=0.5,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.75,
        )
        ax.text(
            (int(rst["start"]) + int(rst["end"])) / 2,
            restock_y,
            f"{rst['line_id']}-{rst['sku']}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
        )

    title = (
        f"{result['_path'].name} | status={result.get('status', 'N/A')} | "
        f"profit={result.get('net_profit', 'N/A')}"
    )
    ax.set_title(title, fontsize=10)
    ax.set_yticks(list(range(len(roasters) + 1)))
    ax.set_yticklabels(roasters + ["RESTOCK"])
    ax.set_xlim(0, 480)
    ax.set_xlabel("Minute of shift")
    ax.set_ylabel("Roaster / shared station")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def add_legend(fig) -> None:
    import matplotlib.patches as mpatches

    handles = [
        mpatches.Patch(facecolor=SKU_COLORS["PSC"], edgecolor="black", label="PSC"),
        mpatches.Patch(facecolor=SKU_COLORS["NDG"], edgecolor="black", label="NDG"),
        mpatches.Patch(facecolor=SKU_COLORS["BUSTA"], edgecolor="black", label="BUSTA"),
        mpatches.Patch(facecolor="#E45756", edgecolor="black", label="Restock L1"),
        mpatches.Patch(facecolor="#B279A2", edgecolor="black", label="Restock L2"),
        mpatches.Patch(facecolor="white", edgecolor="black", hatch="//", label="MTO"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=6, frameon=False)


def main() -> int:
    args = parse_args()

    results = [load_result(path_str) for path_str in args.json_files]
    plot_results(results, save_path=args.save, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
