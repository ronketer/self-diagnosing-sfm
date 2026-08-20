"""Generate the TempleRing ablation figure from saved experiment metrics."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).parent
RUNS = [
    ("Full pipeline", "templeRing"),
    ("No bundle\nadjustment", "templeRing_no_ba"),
    ("No geometric\nverification", "templeRing_no_geom"),
]


def load_metrics(output_name: str) -> dict:
    path = ROOT / "output" / output_name / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run the corresponding experiment from README.md"
        )
    return json.loads(path.read_text())


def plot_pipeline(out_path: Path) -> None:
    """Write a compact overview of the implemented SfM stages."""
    stages = [
        "Input images\n+ known/estimated K",
        "SIFT detection\n+ mutual matching",
        "Essential RANSAC\n+ pair verification",
        "Tracks + robust\n+ initial pair",
        "Incremental PnP\n+ triangulation",
        "Periodic BA\n+ outlier pruning",
        "Camera poses +\n+ sparse point cloud",
        "Alignment +\n+ evaluation",
    ]
    positions = [
        (0.10, 0.70),
        (0.37, 0.70),
        (0.64, 0.70),
        (0.90, 0.70),
        (0.90, 0.28),
        (0.64, 0.28),
        (0.37, 0.28),
        (0.10, 0.28),
    ]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0, 1)
    ax.axis("off")
    for stage, (x, y) in zip(stages, positions):
        ax.text(
            x,
            y,
            stage,
            ha="center",
            va="center",
            fontsize=11,
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#e8f1fb",
                "edgecolor": "#315b7d",
                "linewidth": 1.5,
            },
        )
    for start, end in zip(positions, positions[1:]):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=1.5,
                color="#315b7d",
                shrinkA=58,
                shrinkB=58,
                connectionstyle="arc3,rad=0.0",
            )
        )
    ax.set_title("Incremental Structure-from-Motion pipeline", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    labels = [label for label, _ in RUNS]
    metrics = [load_metrics(name) for _, name in RUNS]
    registered = [100 * m["n_registered"] / m.get("n_images", 47) for m in metrics]
    reprojection = [m["rms_reproj_final_px"] for m in metrics]
    rotation = [m["rotation_error_deg"]["mean"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    panels = [
        (registered, "registered cameras (%)", (0, 105)),
        (reprojection, "RMS reprojection error (px)", None),
        (rotation, "mean rotation error (deg)", None),
    ]
    colors = ["tab:blue", "tab:orange", "tab:red"]
    for ax, (values, title, limits) in zip(axes, panels):
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    fig.tight_layout()

    out_path = ROOT / "report" / "figures" / "ablation_summary.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")

    pipeline_path = out_path.parent / "pipeline_overview.png"
    plot_pipeline(pipeline_path)
    print(f"wrote {pipeline_path}")


if __name__ == "__main__":
    main()
