"""Command-line entry point for the robust incremental SfM project."""

import argparse
import time
from pathlib import Path

import numpy as np

from src.dataset import load_dataset
from src.evaluation import (
    evaluate,
    save_diagnostics_json,
    save_reconstruction_json,
)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="templeSparseRing")
    parser.add_argument("--skip-ba", action="store_true")
    parser.add_argument(
        "--skip-geometric-verification",
        action="store_true",
    )
    parser.add_argument(
        "--output-label",
        help="append a label to the output directory",
    )
    parser.add_argument("--skip-colmap-export", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    data_dir = ROOT / "data" / args.dataset

    output_name = args.dataset
    if args.output_label:
        output_name += f"_{args.output_label}"

    out_dir = ROOT / "output" / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    images, gts, intrinsics = load_dataset(data_dir)

    print(
        f"[{time.time() - start_time:6.1f}s] "
        f"loaded {len(images)} images"
    )

    keypoints = []
    descriptors = []

    for image in images:
        kp, descriptor = detect_features(image.gray)
        keypoints.append(kp)
        descriptors.append(descriptor)

    print(
        f"[{time.time() - start_time:6.1f}s] features: "
        f"{sum(len(kp) for kp in keypoints)} keypoints"
    )

    # Match all pairs.  Image ordering is not assumed to encode view adjacency.
    cache_dir = ROOT / "output" / args.dataset / "cache"
    cache_path = (
        None
        if args.no_cache
        else cache_dir / f"matches_{args.dataset}_all.npz"
    )

    match_graph = build_match_graph(
        descriptors,
        cache_path=cache_path,
    )

    print(
        f"[{time.time() - start_time:6.1f}s] "
        f"match graph: {len(match_graph)} pairs"
    )

    reconstruction, track_manager, stats = run_sfm(
        keypoints,
        match_graph,
        intrinsics,
        use_ba=not args.skip_ba,
        use_geometric_verification=(
            not args.skip_geometric_verification
        ),
    )

    stats["runtime_seconds"] = time.time() - start_time

    print(
        f"[{time.time() - start_time:6.1f}s] SfM done: "
        f"{stats['n_registered']} cams, "
        f"{stats['n_points']} points"
    )

    # Failed initialization is a legitimate diagnostic result.  Persist it
    # cleanly instead of letting plotting/evaluation code crash on an empty
    # reconstruction.
    if stats.get("status") == "FAILED":
        save_diagnostics_json(
            stats,
            out_dir / "diagnostics.json",
        )
        save_reconstruction_json(
            reconstruction,
            stats,
            out_dir / "reconstruction.json",
        )
        print(
            "reconstruction failed during initialization; "
            "diagnostics were written and geometric plotting was skipped"
        )
        return

    metrics = None

    if gts is not None:
        metrics = evaluate(
            reconstruction,
            gts,
            stats,
            out_dir,
        )
        print(
            (out_dir / "metrics.txt").read_text()
        )
    else:
        save_diagnostics_json(
            stats,
            out_dir / "diagnostics.json",
        )
        save_reconstruction_json(
            reconstruction,
            stats,
            out_dir / "reconstruction.json",
        )
        print(
            "no reference poses for this dataset; "
            "skipping pose evaluation"
        )

    # Use the same geometrically verified matches that actually drove
    # initialization.  Do not label raw descriptor matches as inliers.
    if len(stats["registration_order"]) >= 2:
        image_a, image_b = stats["registration_order"][:2]
        verified_matches = np.asarray(
            stats.get(
                "initial_pair_verified_matches",
                np.zeros((0, 2), dtype=np.int64),
            ),
            dtype=np.int64,
        ).reshape(-1, 2)

        if len(verified_matches):
            plot_matches(
                images[image_a],
                images[image_b],
                keypoints[image_a][verified_matches[:, 0]],
                keypoints[image_b][verified_matches[:, 1]],
                np.ones(
                    len(verified_matches),
                    dtype=bool,
                ),
                out_dir / "matches_init_pair.png",
            )

    reprojection_errors = reprojection_error_list(
        reconstruction,
        track_manager,
        keypoints,
    )

    plot_reproj_hist(
        None,
        reprojection_errors,
        out_dir / "reproj_hist.png",
    )

    plot_ba_history(
        stats,
        out_dir / "ba_history.png",
    )

    plot_reconstruction(
        reconstruction,
        track_manager,
        images,
        keypoints,
        gts,
        out_dir / "reconstruction_3d.png",
    )

    if gts is not None and metrics is not None:
        plot_camera_ring(
            reconstruction,
            gts,
            out_dir / "camera_ring.png",
        )
        plot_pose_error_by_registration(
            metrics,
            stats["registration_order"],
            out_dir / "pose_error_by_registration.png",
        )

    export_ply(
        reconstruction,
        track_manager,
        images,
        keypoints,
        out_dir / "pointcloud.ply",
    )

    if not args.skip_colmap_export:
        import cv2

        from src.colmap_export import export_colmap_text
        from src.visualization import point_colors

        colmap_dir = out_dir / "colmap"

        rgb = point_colors(
            track_manager,
            reconstruction,
            images,
            keypoints,
            set(reconstruction.cameras),
        )

        export_colmap_text(
            reconstruction,
            track_manager,
            images,
            keypoints,
            rgb,
            colmap_dir / "sparse" / "0",
        )

        image_output = colmap_dir / "images"
        image_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        for image_id in sorted(reconstruction.cameras):
            cv2.imwrite(
                str(
                    image_output
                    / images[image_id].name
                ),
                cv2.cvtColor(
                    images[image_id].color,
                    cv2.COLOR_RGB2BGR,
                ),
            )

        print(
            f"[{time.time() - start_time:6.1f}s] "
            f"COLMAP text export written to {colmap_dir}"
        )

    print(
        f"[{time.time() - start_time:6.1f}s] "
        f"outputs written to {out_dir}"
    )


if __name__ == "__main__":
    main()
