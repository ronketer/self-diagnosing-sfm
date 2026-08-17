"""Figures and point-cloud export."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import aligned_gt_overlay
from .triangulation import reproject


def point_colors(tm, recon, images, keypoints, registered) -> np.ndarray:
    """Mean observed RGB per 3D point (uint8, (N,3)), sampled at the nearest
    pixel to each observing keypoint."""
    colors = np.zeros((len(recon.points3d), 3))
    counts = np.zeros(len(recon.points3d))
    for tr in tm.tracks:
        if tr.point3d_id is None:
            continue
        for img, kp in tr.obs.items():
            if img not in registered:
                continue
            x, y = np.round(keypoints[img][kp]).astype(int)
            h, w = images[img].color.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                colors[tr.point3d_id] += images[img].color[y, x]
                counts[tr.point3d_id] += 1
    counts[counts == 0] = 1
    return (colors / counts[:, None]).astype(np.uint8)


def plot_reconstruction(recon, tm, images, keypoints, gts, out_path: Path):
    registered = set(recon.poses)
    if gts is not None:
        reg, C_est_al, C_gt, pts_al = aligned_gt_overlay(recon, gts)
        title = "Reconstruction (aligned to GT)"
    else:
        C_est_al = np.array([recon.camera_center(i) for i in sorted(recon.poses)])
        C_gt, pts_al = None, recon.points3d
        title = "Reconstruction"
    colors = point_colors(tm, recon, images, keypoints, registered) / 255.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts_al[:, 0], pts_al[:, 1], pts_al[:, 2], c=colors, s=1)
    ax.scatter(*C_est_al.T, c="red", s=40, marker="^", label="estimated cams")
    if C_gt is not None:
        ax.scatter(*C_gt.T, c="blue", s=40, marker="o", alpha=0.5, label="GT cams")
    ax.legend()
    ax.set_title(title)
    ax.view_init(elev=-70, azim=-90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_camera_ring(recon, gts, out_path: Path):
    reg, C_est_al, C_gt, _ = aligned_gt_overlay(recon, gts)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(C_gt[:, 0], C_gt[:, 2], "bo-", label="GT", alpha=0.6)
    ax.plot(C_est_al[:, 0], C_est_al[:, 2], "r^--", label="estimated")
    for k, i in enumerate(reg):
        ax.annotate(str(i), (C_est_al[k, 0], C_est_al[k, 2]), fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Camera ring, top-down (aligned)")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_matches(img1, img2, pts1, pts2, mask, out_path: Path, max_draw=80):
    h1, w1 = img1.color.shape[:2]
    h2, w2 = img2.color.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1.color
    canvas[:h2, w1:] = img2.color
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)
    idx = np.where(mask)[0]
    if len(idx) > max_draw:
        idx = np.random.default_rng(0).choice(idx, max_draw, replace=False)
    for k in idx:
        ax.plot([pts1[k, 0], pts2[k, 0] + w1], [pts1[k, 1], pts2[k, 1]], "-", lw=0.5)
    ax.set_title(f"{mask.sum()} inlier matches (showing {len(idx)})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reproj_hist(errors_before, errors_after, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 5, 60)
    if errors_before is not None:
        ax.hist(np.clip(errors_before, 0, 5), bins=bins, alpha=0.5, label="before BA")
    ax.hist(np.clip(errors_after, 0, 5), bins=bins, alpha=0.5, label="after BA")
    ax.set_xlabel("reprojection error (px)")
    ax.set_ylabel("observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ba_history(stats: dict, out_path: Path):
    """Plot reprojection RMS immediately before and after each BA call."""
    history = stats.get("ba_history", [])
    if not history:
        return
    x = np.arange(1, len(history) + 1)
    before = [entry["rms_before_px"] for entry in history]
    after = [entry["rms_after_px"] for entry in history]
    labels = [
        f"{entry['n_registered']} cams\n{entry['stage']}" for entry in history
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(history) * 0.6), 5))
    ax.plot(x, before, "o--", label="before BA", color="tab:orange")
    ax.plot(x, after, "o-", label="after BA", color="tab:blue")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("RMS reprojection error (px)")
    ax.set_xlabel("bundle-adjustment invocation")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pose_error_by_registration(metrics: dict, registration_order, out_path: Path):
    """Plot aligned pose error in the order cameras entered the model."""
    rotation = metrics["rotation_error_deg"]["per_camera"]
    center = metrics["camera_center_error"]["per_camera"]
    diameter = metrics["scene_diameter"]
    order = [i for i in registration_order if str(i) in rotation]
    rot_values = [rotation[str(i)] for i in order]
    center_values = [center[str(i)] / diameter * 100 for i in order]

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    steps = np.arange(1, len(order) + 1)
    axes[0].plot(steps, rot_values, "o-", markersize=3)
    axes[0].set_ylabel("rotation error (deg)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(steps, center_values, "o-", markersize=3, color="tab:green")
    axes[1].set_ylabel("center error (% diameter)")
    axes[1].set_xlabel("registration step")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reprojection_error_list(recon, tm, keypoints) -> np.ndarray:
    K = recon.K
    errs = []
    for tr in tm.tracks:
        if tr.point3d_id is None:
            continue
        X = recon.points3d[tr.point3d_id].reshape(1, 3)
        for img, kp in tr.obs.items():
            if img in recon.poses:
                R, t = recon.poses[img]
                errs.append(np.linalg.norm(reproject(K, R, t, X)[0] - keypoints[img][kp]))
    return np.array(errs)


def export_ply(recon, tm, images, keypoints, out_path: Path):
    registered = set(recon.poses)
    colors = point_colors(tm, recon, images, keypoints, registered)
    pts = recon.points3d
    with open(out_path, "w") as f:
        f.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(pts)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "end_header\n"
        )
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]}\n")
