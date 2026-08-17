"""Reconstruction figures, reprojection plots, and PLY export."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation import aligned_gt_overlay
from .geometry.projection import reproject
from .models import Reconstruction
from .tracks import TrackManager


def point_colors(
    track_manager: TrackManager,
    reconstruction: Reconstruction,
    images,
    keypoints,
    registered: set[int] | None = None,
) -> np.ndarray:
    """Mean RGB color for each reconstructed landmark."""

    if registered is None:
        registered = set(reconstruction.cameras)

    colors = np.zeros((len(reconstruction.landmarks), 3), dtype=float)
    counts = np.zeros(len(reconstruction.landmarks), dtype=float)

    for track in track_manager.tracks:
        if track.point3d_id is None:
            continue

        for observation in track.observations.values():
            image_id = observation.image_id
            if image_id not in registered:
                continue

            x, y = np.round(
                keypoints[image_id][observation.keypoint_id]
            ).astype(int)

            height, width = images[image_id].color.shape[:2]
            if 0 <= y < height and 0 <= x < width:
                colors[track.point3d_id] += images[image_id].color[y, x]
                counts[track.point3d_id] += 1

    counts[counts == 0] = 1
    return (colors / counts[:, None]).astype(np.uint8)


def plot_reconstruction(
    reconstruction: Reconstruction,
    track_manager: TrackManager,
    images,
    keypoints,
    gts,
    out_path: Path,
):
    registered = set(reconstruction.cameras)

    if gts is not None:
        _, centers_est, centers_gt, points = aligned_gt_overlay(
            reconstruction,
            gts,
        )
        title = "Reconstruction (aligned to reference)"
    else:
        centers_est = np.asarray(
            [
                reconstruction.cameras[image_id].pose.center
                for image_id in sorted(reconstruction.cameras)
            ]
        )
        centers_gt = None
        points = reconstruction.points3d
        title = "Reconstruction"

    colors = (
        point_colors(
            track_manager,
            reconstruction,
            images,
            keypoints,
            registered,
        )
        / 255.0
    )

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    if len(points):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=1,
        )

    if len(centers_est):
        ax.scatter(
            *centers_est.T,
            c="red",
            s=40,
            marker="^",
            label="estimated cams",
        )

    if centers_gt is not None and len(centers_gt):
        ax.scatter(
            *centers_gt.T,
            c="blue",
            s=40,
            marker="o",
            alpha=0.5,
            label="reference cams",
        )

    ax.legend()
    ax.set_title(title)
    ax.view_init(elev=-70, azim=-90)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_camera_ring(
    reconstruction: Reconstruction,
    gts,
    out_path: Path,
):
    registered, centers_est, centers_gt, _ = aligned_gt_overlay(
        reconstruction,
        gts,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(
        centers_gt[:, 0],
        centers_gt[:, 2],
        "bo-",
        label="reference",
        alpha=0.6,
    )
    ax.plot(
        centers_est[:, 0],
        centers_est[:, 2],
        "r^--",
        label="estimated",
    )

    for index, image_id in enumerate(registered):
        ax.annotate(
            str(image_id),
            (centers_est[index, 0], centers_est[index, 2]),
            fontsize=8,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title("Camera ring, top-down (aligned)")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_matches(
    img1,
    img2,
    pts1,
    pts2,
    mask,
    out_path: Path,
    max_draw: int = 80,
):
    """Draw a subset of already-classified pairwise correspondences."""

    h1, w1 = img1.color.shape[:2]
    h2, w2 = img2.color.shape[:2]

    canvas = np.zeros(
        (max(h1, h2), w1 + w2, 3),
        dtype=np.uint8,
    )
    canvas[:h1, :w1] = img1.color
    canvas[:h2, w1:] = img2.color

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(canvas)

    indices = np.where(mask)[0]
    if len(indices) > max_draw:
        indices = np.random.default_rng(0).choice(
            indices,
            max_draw,
            replace=False,
        )

    for match_id in indices:
        ax.plot(
            [
                pts1[match_id, 0],
                pts2[match_id, 0] + w1,
            ],
            [
                pts1[match_id, 1],
                pts2[match_id, 1],
            ],
            "-",
            lw=0.5,
        )

    ax.set_title(
        f"{int(np.asarray(mask).sum())} verified matches "
        f"(showing {len(indices)})"
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reproj_hist(
    errors_before,
    errors_after,
    out_path: Path,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 5, 60)

    if errors_before is not None:
        ax.hist(
            np.clip(errors_before, 0, 5),
            bins=bins,
            alpha=0.5,
            label="before BA",
        )

    ax.hist(
        np.clip(errors_after, 0, 5),
        bins=bins,
        alpha=0.5,
        label="after BA",
    )

    ax.set_xlabel("reprojection error (px)")
    ax.set_ylabel("observations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_ba_history(
    stats: dict,
    out_path: Path,
):
    history = stats.get("ba_history", [])
    if not history:
        return

    x = np.arange(1, len(history) + 1)
    before = [entry["rms_before_px"] for entry in history]
    after = [entry["rms_after_px"] for entry in history]
    labels = [
        f"{entry['n_registered']} cams\n{entry['stage']}"
        for entry in history
    ]

    fig, ax = plt.subplots(
        figsize=(max(8, len(history) * 0.6), 5)
    )
    ax.plot(x, before, "o--", label="before BA")
    ax.plot(x, after, "o-", label="after BA")
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("RMS reprojection error (px)")
    ax.set_xlabel("bundle-adjustment invocation")
    ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pose_error_by_registration(
    metrics: dict,
    registration_order,
    out_path: Path,
):
    rotation = metrics["rotation_error_deg"]["per_camera"]
    center = metrics["camera_center_error"]["per_camera"]
    diameter = metrics["scene_diameter"]

    order = [
        image_id
        for image_id in registration_order
        if str(image_id) in rotation
    ]

    rotation_values = [
        rotation[str(image_id)]
        for image_id in order
    ]
    center_values = [
        center[str(image_id)] / diameter * 100
        for image_id in order
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 6),
        sharex=True,
    )

    steps = np.arange(1, len(order) + 1)

    axes[0].plot(
        steps,
        rotation_values,
        "o-",
        markersize=3,
    )
    axes[0].set_ylabel("rotation error (deg)")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        steps,
        center_values,
        "o-",
        markersize=3,
    )
    axes[1].set_ylabel("center error (% diameter)")
    axes[1].set_xlabel("registration step")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reprojection_error_list(
    reconstruction: Reconstruction,
    track_manager: TrackManager,
    keypoints,
) -> np.ndarray:
    errors: list[float] = []

    for track in track_manager.tracks:
        if track.point3d_id is None:
            continue

        landmark = reconstruction.landmarks[track.point3d_id]

        for observation in track.observations.values():
            if observation.image_id not in reconstruction.cameras:
                continue

            pose = reconstruction.cameras[observation.image_id].pose
            projected = reproject(
                reconstruction.K,
                pose.R,
                pose.t,
                landmark.xyz.reshape(1, 3),
            )[0]

            measured = keypoints[
                observation.image_id
            ][observation.keypoint_id]

            errors.append(
                float(np.linalg.norm(projected - measured))
            )

    return np.asarray(errors)


def export_ply(
    reconstruction: Reconstruction,
    track_manager: TrackManager,
    images,
    keypoints,
    out_path: Path,
):
    colors = point_colors(
        track_manager,
        reconstruction,
        images,
        keypoints,
        set(reconstruction.cameras),
    )
    points = reconstruction.points3d

    with open(out_path, "w") as file:
        file.write(
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )

        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )
