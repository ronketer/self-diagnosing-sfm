"""DLT triangulation with explicit geometric validation."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .config import SfMConfig
from .geometry.projection import (
    camera_center,
    point_depths,
    reproject,
    to_normalized,
    triangulation_angles_deg,
)
from .models import CameraIntrinsics, CameraPose, RegisteredCamera, Track
from .validation.diagnostics import FailureReason


@dataclass
class TriangulationResult:
    """Result of triangulating one multi-view feature track."""

    accepted: bool
    xyz: np.ndarray | None = None
    parallax_deg: float | None = None
    best_parallax_pair: tuple[int, int] | None = None
    max_reprojection_error_px: float | None = None
    reason: FailureReason | str | None = None


def triangulate_two_view(
    P1: np.ndarray,
    P2: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
) -> np.ndarray:
    """Vectorized two-view DLT.

    ``P1`` and ``P2`` are normalized 3x4 projection matrices.
    ``x1`` and ``x2`` are normalized image coordinates of shape ``(N,2)``.
    """

    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.ndim == 1:
        x1 = x1.reshape(1, 2)
    if x2.ndim == 1:
        x2 = x2.reshape(1, 2)

    n = len(x1)
    A = np.empty((n, 4, 4), dtype=float)
    A[:, 0] = x1[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = x1[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = x2[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = x2[:, 1:2] * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[:, -1]
    w = X_h[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return X_h[:, :3] / w


def triangulate_multi_view(
    P_list: list[np.ndarray],
    x_list: list[np.ndarray],
) -> np.ndarray:
    """Multi-view linear DLT for one point observed in at least two views."""

    if len(P_list) != len(x_list) or len(P_list) < 2:
        raise ValueError("Multi-view triangulation requires >=2 matching views.")

    rows = []
    for P, x in zip(P_list, x_list):
        rows.append(x[0] * P[2] - P[0])
        rows.append(x[1] * P[2] - P[1])

    _, _, Vt = np.linalg.svd(np.asarray(rows, dtype=float))
    X_h = Vt[-1]
    w = X_h[3] if abs(X_h[3]) > 1e-12 else 1e-12
    return X_h[:3] / w


def _pose_from_value(
    value: RegisteredCamera | CameraPose | tuple[np.ndarray, np.ndarray],
) -> CameraPose:
    if isinstance(value, RegisteredCamera):
        return value.pose
    if isinstance(value, CameraPose):
        return value
    return CameraPose(R=value[0], t=value[1])


def _observation_indices(track_or_obs: Track | Mapping[int, int]) -> dict[int, int]:
    """Normalize typed Track or legacy observation mapping to image->keypoint ID."""

    if isinstance(track_or_obs, Track):
        return {
            image_id: observation.keypoint_id
            for image_id, observation in track_or_obs.observations.items()
        }
    return {int(image_id): int(kp_id) for image_id, kp_id in track_or_obs.items()}


def triangulate_track(
    track_or_obs: Track | Mapping[int, int],
    cameras_or_poses: Mapping[
        int,
        RegisteredCamera | CameraPose | tuple[np.ndarray, np.ndarray],
    ],
    keypoints: list[np.ndarray],
    intrinsics: CameraIntrinsics | np.ndarray,
    config: SfMConfig | None = None,
) -> TriangulationResult:
    """Triangulate one track and validate depth, parallax, and reprojection.

    Production code passes a typed :class:`Track` and the reconstruction's
    registered-camera dictionary.  Mapping-based inputs remain supported as a
    numerical/test adapter.
    """

    if config is None:
        config = SfMConfig()

    K = intrinsics.K if isinstance(intrinsics, CameraIntrinsics) else np.asarray(intrinsics, dtype=float)
    obs = _observation_indices(track_or_obs)

    views = sorted(image_id for image_id in obs if image_id in cameras_or_poses)
    if len(views) < 2:
        return TriangulationResult(
            accepted=False,
            reason=FailureReason.INSUFFICIENT_MATCHES,
        )

    P_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    poses: dict[int, CameraPose] = {}

    for image_id in views:
        pose = _pose_from_value(cameras_or_poses[image_id])
        poses[image_id] = pose
        P_list.append(np.hstack([pose.R, pose.t.reshape(3, 1)]))

        pixel = keypoints[image_id][obs[image_id]]
        normalized = to_normalized(pixel.reshape(1, 2), K)[0]
        x_list.append(normalized)
        centers.append(camera_center(pose.R, pose.t))

    X = triangulate_multi_view(P_list, x_list)

    if not np.all(np.isfinite(X)):
        return TriangulationResult(
            accepted=False,
            reason=FailureReason.NONFINITE_TRIANGULATION,
        )

    # 1. Cheirality: the landmark must lie in front of every registered
    # camera currently contributing to this triangulation.
    for image_id in views:
        pose = poses[image_id]
        depth = point_depths(pose.R, pose.t, X.reshape(1, 3))[0]
        if depth <= 0:
            return TriangulationResult(
                accepted=False,
                xyz=X,
                reason=FailureReason.NEGATIVE_DEPTH,
            )

    # 2. Parallax: use the strongest baseline among all observing camera pairs,
    # never an arbitrary first/last pair.
    max_parallax = 0.0
    best_pair: tuple[int, int] | None = None

    for (center_idx_a, view_a), (center_idx_b, view_b) in itertools.combinations(
        enumerate(views), 2
    ):
        angle = float(
            triangulation_angles_deg(
                centers[center_idx_a],
                centers[center_idx_b],
                X.reshape(1, 3),
            )[0]
        )

        if best_pair is None or angle > max_parallax:
            max_parallax = angle
            best_pair = (view_a, view_b)

    if max_parallax < config.min_parallax_deg:
        return TriangulationResult(
            accepted=False,
            xyz=X,
            parallax_deg=max_parallax,
            best_parallax_pair=best_pair,
            reason=FailureReason.LOW_PARALLAX,
        )

    # 3. Reprojection: reject if any participating observation has excessive
    # pixel error.
    max_error = 0.0

    for image_id in views:
        pose = poses[image_id]
        projected = reproject(
            K,
            pose.R,
            pose.t,
            X.reshape(1, 3),
        )[0]

        error = float(
            np.linalg.norm(
                projected - keypoints[image_id][obs[image_id]]
            )
        )
        max_error = max(max_error, error)

        if error > config.max_triangulation_error_px:
            return TriangulationResult(
                accepted=False,
                xyz=X,
                parallax_deg=max_parallax,
                best_parallax_pair=best_pair,
                max_reprojection_error_px=max_error,
                reason=FailureReason.HIGH_REPROJECTION_ERROR,
            )

    return TriangulationResult(
        accepted=True,
        xyz=X,
        parallax_deg=max_parallax,
        best_parallax_pair=best_pair,
        max_reprojection_error_px=max_error,
        reason=None,
    )
