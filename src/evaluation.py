"""Evaluation against Middlebury ground truth."""

import json
from pathlib import Path

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t) minimizing ||s R src + t - dst||^2."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (sc**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return s, R, t


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = np.clip((np.trace(R1 @ R2.T) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def evaluate(recon, gts, stats: dict, out_dir: Path) -> dict:
    """Align estimated camera centers to GT via Umeyama, report per-camera
    rotation and center errors. Writes metrics.json / metrics.txt."""
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

    out_dir.mkdir(parents=True, exist_ok=True)
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
