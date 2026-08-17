"""Evaluation against reference camera poses and machine-readable reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry.pose import rotation_error_deg, umeyama
from .models import Reconstruction


def save_reconstruction_json(
    reconstruction: Reconstruction,
    stats: dict,
    out_path: Path,
) -> None:
    """Serialize calibrated cameras, landmarks, and summary information."""

    cameras = []
    for image_id in sorted(reconstruction.cameras):
        camera = reconstruction.cameras[image_id]
        pose = camera.pose
        cameras.append(
            {
                "image_id": int(image_id),
                "R": pose.R.tolist(),
                "t": pose.t.tolist(),
                "center": pose.center.tolist(),
            }
        )

    data = {
        "intrinsics": {
            "K": reconstruction.intrinsics.K.tolist(),
            "distortion": (
                None
                if reconstruction.intrinsics.distortion is None
                else reconstruction.intrinsics.distortion.tolist()
            ),
        },
        "registration_order": [
            int(image_id)
            for image_id in stats.get("registration_order", [])
        ],
        "registered_cameras": cameras,
        "n_points": len(reconstruction.landmarks),
        "landmarks": [
            {
                "id": int(landmark.id),
                "xyz": landmark.xyz.tolist(),
                "track_id": int(landmark.track_id),
                "reprojection_error_px": landmark.reprojection_error_px,
            }
            for landmark in reconstruction.landmarks
        ],
        "reprojection_rms_px": stats.get("rms_final"),
        "status": stats.get("status"),
        "termination_reason": stats.get("termination_reason"),
    }

    out_path.write_text(json.dumps(data, indent=2))


def save_diagnostics_json(
    stats: dict,
    out_path: Path,
) -> None:
    """Save structured diagnostic events and final camera statuses."""

    diagnostics = stats.get("diagnostics")

    if diagnostics is not None and hasattr(diagnostics, "to_dict"):
        data = diagnostics.to_dict()
    else:
        init_pair = stats.get("registration_order", [])[:2]
        data = {
            "status": stats.get("status"),
            "termination_reason": stats.get("termination_reason"),
            "summary": {
                "initial_pair": [int(x) for x in init_pair],
                "registered_images": int(stats.get("n_registered", 0)),
                "total_images": int(stats.get("n_images", 0)),
                "landmarks": int(stats.get("n_points", 0)),
                "final_reprojection_rms_px": stats.get("rms_final"),
            },
            "rejections": [],
            "initial_pair_evaluations": [],
            "events": [],
        }

    out_path.write_text(json.dumps(data, indent=2))


def evaluate(
    reconstruction: Reconstruction,
    gts,
    stats: dict,
    out_dir: Path,
) -> dict:
    """Align estimated camera centers to reference poses and report errors.

    A Sim(3) alignment is required because monocular calibrated SfM recovers
    the scene only up to a global similarity transformation.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    save_diagnostics_json(stats, out_dir / "diagnostics.json")
    save_reconstruction_json(
        reconstruction,
        stats,
        out_dir / "reconstruction.json",
    )

    registered = sorted(reconstruction.cameras)
    if len(registered) < 2:
        raise ValueError(
            "Pose evaluation requires at least two registered cameras."
        )

    C_est = np.asarray(
        [
            reconstruction.cameras[image_id].pose.center
            for image_id in registered
        ]
    )
    C_gt = np.asarray([gts[image_id].center for image_id in registered])
    C_gt_all = np.asarray([gt.center for gt in gts])

    scale, R_align, t_align = umeyama(C_est, C_gt)
    C_est_aligned = (
        scale * (R_align @ C_est.T)
    ).T + t_align

    diameter = float(
        np.linalg.norm(
            C_gt_all.max(axis=0) - C_gt_all.min(axis=0)
        )
    )
    center_error = np.linalg.norm(C_est_aligned - C_gt, axis=1)

    rotation_error = []
    for image_id in registered:
        R_est = reconstruction.cameras[image_id].pose.R
        rotation_error.append(
            rotation_error_deg(
                R_est @ R_align.T,
                gts[image_id].R,
            )
        )
    rotation_error = np.asarray(rotation_error)

    metrics = {
        "n_registered": len(registered),
        "n_images": int(stats.get("n_images", len(gts))),
        "registration_ratio": float(
            len(registered) / stats.get("n_images", len(gts))
        ),
        "n_points": len(reconstruction.landmarks),
        "n_observations": int(stats.get("n_observations", 0)),
        "mean_track_length": float(
            stats.get("mean_track_length", 0.0)
        ),
        "scene_diameter": diameter,
        "rotation_error_deg": {
            "per_camera": {
                str(image_id): float(error)
                for image_id, error in zip(
                    registered,
                    rotation_error,
                )
            },
            "mean": float(rotation_error.mean()),
            "median": float(np.median(rotation_error)),
            "p90": float(np.percentile(rotation_error, 90)),
            "max": float(rotation_error.max()),
        },
        "camera_center_error": {
            "per_camera": {
                str(image_id): float(error)
                for image_id, error in zip(
                    registered,
                    center_error,
                )
            },
            "mean": float(center_error.mean()),
            "mean_pct_of_diameter": float(
                center_error.mean() / diameter * 100
            ),
            "p90_pct_of_diameter": float(
                np.percentile(
                    center_error / diameter * 100,
                    90,
                )
            ),
            "max": float(center_error.max()),
        },
        "rms_reproj_before_final_ba_px": stats.get(
            "rms_before_final_ba"
        ),
        "rms_reproj_final_px": stats.get("rms_final"),
        "runtime_seconds": stats.get("runtime_seconds"),
        "status": stats.get("status"),
        "termination_reason": stats.get("termination_reason"),
        "configuration": {
            "bundle_adjustment": bool(
                stats.get("use_ba", True)
            ),
            "geometric_verification": bool(
                stats.get("use_geometric_verification", True)
            ),
        },
        "matching": {
            "raw_pairs": int(stats.get("raw_match_pairs", 0)),
            "raw_matches": int(stats.get("raw_matches", 0)),
            "verified_pairs": int(
                stats.get("verified_match_pairs", 0)
            ),
            "verified_matches": int(
                stats.get("verified_matches", 0)
            ),
        },
        "ba_history": stats.get("ba_history", []),
        "alignment": {
            "scale": float(scale),
        },
    }

    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    lines = [
        f"Status             : {metrics['status']}",
        f"Termination        : {metrics['termination_reason']}",
        f"Registered cameras : {metrics['n_registered']}/{metrics['n_images']} "
        f"({metrics['registration_ratio'] * 100:.1f}%)",
        f"3D points          : {metrics['n_points']}",
        f"Observations       : {metrics['n_observations']} "
        f"(mean track length {metrics['mean_track_length']:.2f})",
        (
            f"RMS reproj (final) : "
            f"{metrics['rms_reproj_final_px']:.3f} px"
            if metrics["rms_reproj_final_px"] is not None
            else "RMS reproj (final) : n/a"
        ),
        (
            f"Runtime            : "
            f"{metrics['runtime_seconds']:.1f} s"
            if metrics["runtime_seconds"] is not None
            else "Runtime            : n/a"
        ),
        "",
        f"{'cam':>4} {'rot err (deg)':>14} "
        f"{'center err':>12} {'% diam':>8}",
    ]

    for index, image_id in enumerate(registered):
        lines.append(
            f"{image_id:>4} "
            f"{rotation_error[index]:>14.3f} "
            f"{center_error[index]:>12.5f} "
            f"{center_error[index] / diameter * 100:>7.2f}%"
        )

    lines += [
        "",
        "rotation error   "
        f"mean {rotation_error.mean():.3f}  "
        f"median {np.median(rotation_error):.3f}  "
        f"p90 {np.percentile(rotation_error, 90):.3f}  "
        f"max {rotation_error.max():.3f} deg",
        "center error     "
        f"mean {center_error.mean():.5f} "
        f"({center_error.mean() / diameter * 100:.2f}% of diameter)",
    ]

    (out_dir / "metrics.txt").write_text("\n".join(lines))
    return metrics


def aligned_gt_overlay(
    reconstruction: Reconstruction,
    gts,
):
    """Return estimated geometry aligned into the reference coordinate frame."""

    registered = sorted(reconstruction.cameras)

    C_est = np.asarray(
        [
            reconstruction.cameras[image_id].pose.center
            for image_id in registered
        ]
    )
    C_gt = np.asarray(
        [gts[image_id].center for image_id in registered]
    )

    scale, R_align, t_align = umeyama(C_est, C_gt)

    C_est_aligned = (
        scale * (R_align @ C_est.T)
    ).T + t_align

    points = reconstruction.points3d
    if len(points):
        points_aligned = (
            scale * (R_align @ points.T)
        ).T + t_align
    else:
        points_aligned = points.copy()

    return (
        registered,
        C_est_aligned,
        C_gt,
        points_aligned,
    )
