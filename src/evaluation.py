"""Evaluation against Middlebury ground truth and quality reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from .geometry.pose import rotation_error_deg, umeyama


def save_reconstruction_json(recon, stats: dict, out_path: Path) -> None:
    """Save camera poses, intrinsics, and summary into a machine-readable JSON."""
    reg = sorted(recon.poses)
    cameras = []
    for i in reg:
        R, t = recon.poses[i]
        C = -R.T @ t
        cameras.append(
            {
                "image_id": int(i),
                "R": R.tolist(),
                "t": t.tolist(),
                "center": C.tolist(),
            }
        )

    data = {
        "intrinsics": {"K": recon.K.tolist()},
        "registration_order": [int(x) for x in stats.get("registration_order", [])],
        "registered_cameras": cameras,
        "n_points": int(len(recon.points3d)),
        "reprojection_rms_px": stats.get("rms_final"),
    }
    out_path.write_text(json.dumps(data, indent=2))


def save_diagnostics_json(stats: dict, out_path: Path) -> None:
    """Save detailed diagnostic events and rejection causes into diagnostics.json."""
    diag = stats.get("diagnostics")
    if diag is not None and hasattr(diag, "to_dict"):
        data = diag.to_dict()
    else:
        init_pair = stats.get("registration_order", [0, 1])[:2]
        data = {
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


def evaluate(recon, gts, stats: dict, out_dir: Path) -> dict:
    """Align estimated camera centers to GT via Umeyama, report per-camera
    rotation and center errors. Writes metrics.json / metrics.txt / diagnostics.json / reconstruction.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    save_diagnostics_json(stats, out_dir / "diagnostics.json")
    save_reconstruction_json(recon, stats, out_dir / "reconstruction.json")

    reg = sorted(recon.poses)
    C_est = np.array([recon.camera_center(i) for i in reg])
    C_gt = np.array([gts[i].center for i in reg])
    C_gt_all = np.array([gt.center for gt in gts])

    s, R_align, t_align = umeyama(C_est, C_gt)
    C_est_al = (s * (R_align @ C_est.T)).T + t_align

    # Normalize against the full input trajectory so partial reconstructions
    # remain comparable to runs that register every camera.
    diam = np.linalg.norm(C_gt_all.max(axis=0) - C_gt_all.min(axis=0))
    center_err = np.linalg.norm(C_est_al - C_gt, axis=1)

    rot_err = []
    for k, i in enumerate(reg):
        R_est, _ = recon.poses[i]
        # world frames differ by R_align: R_est_aligned = R_est @ R_align^T
        rot_err.append(rotation_error_deg(R_est @ R_align.T, gts[i].R))
    rot_err = np.array(rot_err)

    metrics = {
        "n_registered": len(reg),
        "n_images": int(stats.get("n_images", len(gts))),
        "registration_ratio": float(len(reg) / stats.get("n_images", len(gts))),
        "n_points": int(len(recon.points3d)),
        "n_observations": int(stats.get("n_observations", 0)),
        "mean_track_length": float(stats.get("mean_track_length", 0.0)),
        "scene_diameter": float(diam),
        "rotation_error_deg": {
            "per_camera": {str(i): float(e) for i, e in zip(reg, rot_err)},
            "mean": float(rot_err.mean()),
            "median": float(np.median(rot_err)),
            "p90": float(np.percentile(rot_err, 90)),
            "max": float(rot_err.max()),
        },
        "camera_center_error": {
            "per_camera": {str(i): float(e) for i, e in zip(reg, center_err)},
            "mean": float(center_err.mean()),
            "mean_pct_of_diameter": float(center_err.mean() / diam * 100),
            "p90_pct_of_diameter": float(np.percentile(center_err / diam * 100, 90)),
            "max": float(center_err.max()),
        },
        "rms_reproj_before_final_ba_px": stats.get("rms_before_final_ba"),
        "rms_reproj_final_px": stats.get("rms_final"),
        "runtime_seconds": stats.get("runtime_seconds"),
        "configuration": {
            "bundle_adjustment": bool(stats.get("use_ba", True)),
            "geometric_verification": bool(
                stats.get("use_geometric_verification", True)
            ),
        },
        "matching": {
            "raw_pairs": int(stats.get("raw_match_pairs", 0)),
            "raw_matches": int(stats.get("raw_matches", 0)),
            "verified_pairs": int(stats.get("verified_match_pairs", 0)),
            "verified_matches": int(stats.get("verified_matches", 0)),
        },
        "ba_history": stats.get("ba_history", []),
        "alignment": {"scale": float(s)},
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        f"Registered cameras : {metrics['n_registered']}/{metrics['n_images']} "
        f"({metrics['registration_ratio'] * 100:.1f}%)",
        f"3D points          : {metrics['n_points']}",
        f"Observations       : {metrics['n_observations']} "
        f"(mean track length {metrics['mean_track_length']:.2f})",
        f"RMS reproj (final) : {metrics['rms_reproj_final_px']:.3f} px"
        if metrics["rms_reproj_final_px"] is not None
        else "RMS reproj: n/a (BA skipped)",
        f"Runtime            : {metrics['runtime_seconds']:.1f} s"
        if metrics["runtime_seconds"] is not None
        else "Runtime            : n/a",
        "",
        f"{'cam':>4} {'rot err (deg)':>14} {'center err':>12} {'% diam':>8}",
    ]
    for k, i in enumerate(reg):
        lines.append(
            f"{i:>4} {rot_err[k]:>14.3f} {center_err[k]:>12.5f} "
            f"{center_err[k] / diam * 100:>7.2f}%"
        )
    lines += [
        "",
        f"rotation error   mean {rot_err.mean():.3f}  median {np.median(rot_err):.3f}  "
        f"p90 {np.percentile(rot_err, 90):.3f}  max {rot_err.max():.3f} deg",
        f"center error     mean {center_err.mean():.5f} "
        f"({center_err.mean() / diam * 100:.2f}% of diameter)",
    ]
    (out_dir / "metrics.txt").write_text("\n".join(lines))
    return metrics


def aligned_gt_overlay(recon, gts):
    """Return GT centers/rotations mapped into the estimated frame's alignment,
    i.e. estimated centers aligned to GT (for plotting est vs gt together)."""
    reg = sorted(recon.poses)
    C_est = np.array([recon.camera_center(i) for i in reg])
    C_gt = np.array([gts[i].center for i in reg])
    s, R_align, t_align = umeyama(C_est, C_gt)
    C_est_al = (s * (R_align @ C_est.T)).T + t_align
    pts_al = (s * (R_align @ recon.points3d.T)).T + t_align
    return reg, C_est_al, C_gt, pts_al
