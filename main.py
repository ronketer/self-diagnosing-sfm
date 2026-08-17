"""Incremental Structure-from-Motion on the Middlebury temple dataset."""

import argparse
import time
from pathlib import Path

from src.dataset import load_dataset
from src.evaluation import evaluate, save_diagnostics_json, save_reconstruction_json
from src.features import build_match_graph, detect_features
from src.sfm import run_sfm
from src.visualization import (
    export_ply,
    plot_ba_history,
    plot_camera_ring,
    plot_matches,
    plot_pose_error_by_registration,
    plot_reconstruction,
    plot_reproj_hist,
    reprojection_error_list,
)

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="templeSparseRing")
    ap.add_argument("--skip-ba", action="store_true")
    ap.add_argument("--skip-geometric-verification", action="store_true")
    ap.add_argument(
        "--output-label",
        help="append a label to the output directory, e.g. 'no_ba' for an ablation",
    )
    ap.add_argument("--skip-colmap-export", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    data_dir = ROOT / "data" / args.dataset
    output_name = args.dataset
    if args.output_label:
        output_name += f"_{args.output_label}"
    out_dir = ROOT / "output" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    images, gts, K = load_dataset(data_dir)
    print(f"[{time.time() - t0:6.1f}s] loaded {len(images)} images")

    keypoints, descriptors = [], []
    for im in images:
        kp, desc = detect_features(im.gray)
        keypoints.append(kp)
        descriptors.append(desc)
    print(
        f"[{time.time() - t0:6.1f}s] features: "
        f"{sum(len(k) for k in keypoints)} keypoints"
    )

    # match all pairs; geometric verification in run_sfm prunes false matches
    # (image order is not a reliable proxy for view adjacency in this dataset)
    cache_dir = ROOT / "output" / args.dataset / "cache"
    cache = None if args.no_cache else cache_dir / f"matches_{args.dataset}_all.npz"
    graph = build_match_graph(keypoints, descriptors, cache_path=cache)
    print(f"[{time.time() - t0:6.1f}s] match graph: {len(graph)} pairs")

    recon, tm, stats = run_sfm(
        keypoints,
        graph,
        K,
        use_ba=not args.skip_ba,
        use_geometric_verification=not args.skip_geometric_verification,
    )
    stats["runtime_seconds"] = time.time() - t0
    print(
        f"[{time.time() - t0:6.1f}s] SfM done: "
        f"{stats['n_registered']} cams, {stats['n_points']} points"
    )

    if gts is not None:
        metrics = evaluate(recon, gts, stats, out_dir)
        print((out_dir / "metrics.txt").read_text())
    else:
        save_diagnostics_json(stats, out_dir / "diagnostics.json")
        save_reconstruction_json(recon, stats, out_dir / "reconstruction.json")
        print("no ground truth for this dataset; skipping pose evaluation")

    # figures
    i0, j0 = stats["registration_order"][:2]
    m = graph.get((min(i0, j0), max(i0, j0)))
    if m is not None:
        import numpy as np

        plot_matches(
            images[min(i0, j0)], images[max(i0, j0)],
            keypoints[min(i0, j0)][m[:, 0]], keypoints[max(i0, j0)][m[:, 1]],
            np.ones(len(m), dtype=bool), out_dir / "matches_init_pair.png",
        )
    errs = reprojection_error_list(recon, tm, keypoints)
    plot_reproj_hist(None, errs, out_dir / "reproj_hist.png")
    plot_ba_history(stats, out_dir / "ba_history.png")
    plot_reconstruction(
        recon, tm, images, keypoints, gts, out_dir / "reconstruction_3d.png"
    )
    if gts is not None:
        plot_camera_ring(recon, gts, out_dir / "camera_ring.png")
        plot_pose_error_by_registration(
            metrics,
            stats["registration_order"],
            out_dir / "pose_error_by_registration.png",
        )
    export_ply(recon, tm, images, keypoints, out_dir / "pointcloud.ply")

    # Optional standards-based export for interoperability with COLMAP tools.
    if not args.skip_colmap_export:
        import cv2

        from src.colmap_export import export_colmap_text
        from src.visualization import point_colors

        colmap_dir = out_dir / "colmap"
        rgb = point_colors(tm, recon, images, keypoints, set(recon.poses))
        export_colmap_text(
            recon, tm, images, keypoints, rgb, colmap_dir / "sparse" / "0"
        )
        img_out = colmap_dir / "images"
        img_out.mkdir(parents=True, exist_ok=True)
        for i in sorted(recon.poses):
            cv2.imwrite(
                str(img_out / images[i].name),
                cv2.cvtColor(images[i].color, cv2.COLOR_RGB2BGR),
            )
        print(f"[{time.time() - t0:6.1f}s] COLMAP text export written to {colmap_dir}")
    print(f"[{time.time() - t0:6.1f}s] outputs written to {out_dir}")


if __name__ == "__main__":
    main()
