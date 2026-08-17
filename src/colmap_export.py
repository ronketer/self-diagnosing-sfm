"""Export a typed reconstruction to COLMAP's standard text model format."""

from pathlib import Path

import numpy as np

from .models import Reconstruction
from .tracks import TrackManager


def R_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP quaternion ``(w, x, y, z)``."""

    trace = np.trace(R)

    if trace > 0:
        w = np.sqrt(1 + trace) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3

        scale = np.sqrt(
            1 + R[i, i] - R[j, j] - R[k, k]
        ) * 2

        q = np.empty(4)
        q[1 + i] = scale / 4
        q[0] = (R[k, j] - R[j, k]) / scale
        q[1 + j] = (R[j, i] + R[i, j]) / scale
        q[1 + k] = (R[k, i] + R[i, k]) / scale
        w, x, y, z = q

    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def export_colmap_text(
    reconstruction: Reconstruction,
    track_manager: TrackManager,
    images,
    keypoints,
    point_rgb: np.ndarray,
    out_dir: Path,
):
    """Write ``cameras.txt``, ``images.txt`` and ``points3D.txt``."""

    out_dir.mkdir(parents=True, exist_ok=True)

    registered = sorted(reconstruction.cameras)
    if not registered:
        return

    K = reconstruction.intrinsics.K
    height, width = images[registered[0]].gray.shape

    with open(out_dir / "cameras.txt", "w") as file:
        file.write(
            "# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        )
        file.write(
            f"1 PINHOLE {width} {height} "
            f"{K[0, 0]:.6f} {K[1, 1]:.6f} "
            f"{K[0, 2]:.6f} {K[1, 2]:.6f}\n"
        )

    colmap_image_id = {
        image_id: index + 1
        for index, image_id in enumerate(registered)
    }

    observations_per_image = {
        image_id: []
        for image_id in registered
    }

    tracks_per_point: dict[
        int,
        list[tuple[int, int]],
    ] = {}

    for track in track_manager.tracks:
        if track.point3d_id is None:
            continue

        for observation in track.observations.values():
            image_id = observation.image_id
            if image_id not in colmap_image_id:
                continue

            point2d_index = len(
                observations_per_image[image_id]
            )

            x, y = keypoints[image_id][
                observation.keypoint_id
            ]

            observations_per_image[image_id].append(
                (
                    x,
                    y,
                    track.point3d_id,
                )
            )

            tracks_per_point.setdefault(
                track.point3d_id,
                [],
            ).append(
                (
                    colmap_image_id[image_id],
                    point2d_index,
                )
            )

    with open(out_dir / "images.txt", "w") as file:
        file.write(
            "# Image list: IMAGE_ID, QW, QX, QY, QZ, "
            "TX, TY, TZ, CAMERA_ID, NAME\n"
        )
        file.write(
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        )

        for image_id in registered:
            pose = reconstruction.cameras[image_id].pose
            quaternion = R_to_qvec(pose.R)

            file.write(
                f"{colmap_image_id[image_id]} "
                f"{quaternion[0]:.9f} "
                f"{quaternion[1]:.9f} "
                f"{quaternion[2]:.9f} "
                f"{quaternion[3]:.9f} "
                f"{pose.t[0]:.9f} "
                f"{pose.t[1]:.9f} "
                f"{pose.t[2]:.9f} "
                f"1 {images[image_id].name}\n"
            )

            file.write(
                " ".join(
                    f"{x:.3f} {y:.3f} {point_id + 1}"
                    for x, y, point_id
                    in observations_per_image[image_id]
                )
                + "\n"
            )

    with open(out_dir / "points3D.txt", "w") as file:
        file.write(
            "# 3D point list: POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        )

        for landmark in reconstruction.landmarks:
            track = tracks_per_point.get(
                landmark.id,
                [],
            )

            if not track:
                continue

            red, green, blue = point_rgb[landmark.id]
            error = (
                0.0
                if landmark.reprojection_error_px is None
                else landmark.reprojection_error_px
            )

            file.write(
                f"{landmark.id + 1} "
                f"{landmark.xyz[0]:.9f} "
                f"{landmark.xyz[1]:.9f} "
                f"{landmark.xyz[2]:.9f} "
                f"{red} {green} {blue} "
                f"{error:.6f} "
                + " ".join(
                    f"{image_id} {point2d_index}"
                    for image_id, point2d_index in track
                )
                + "\n"
            )
