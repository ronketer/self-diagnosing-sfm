"""Production Perspective-n-Point camera pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import SfMConfig
from .geometry.projection import reproject
from .models import CameraIntrinsics, CameraPose
from .validation.diagnostics import FailureReason


@dataclass
class PnPResult:
    """Result of robust 2D-3D camera pose estimation."""

    accepted: bool
    pose: CameraPose | None
    inlier_indices: np.ndarray
    reprojection_rms_px: float | None
    inlier_ratio: float = 0.0
    reason: FailureReason | str | None = None


def estimate_camera_pose(
    points3d: np.ndarray,
    points2d: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray,
    config: SfMConfig | None = None,
) -> PnPResult:
    """Estimate a world-to-camera pose from 2D-3D correspondences.

    Production pose estimation uses OpenCV's PnP RANSAC followed by
    Levenberg-Marquardt refinement on the RANSAC inliers.

    The accepted pose follows the project-wide convention:

        X_camera = R @ X_world + t

    A pose is accepted only if it satisfies the configured:
    - minimum correspondence count;
    - minimum RANSAC inlier count;
    - minimum inlier ratio;
    - maximum inlier reprojection RMS.

    Parameters
    ----------
    points3d:
        World-space landmark coordinates with shape ``(N, 3)``.

    points2d:
        Corresponding image pixel coordinates with shape ``(N, 2)``.

    intrinsics:
        Shared calibrated pinhole intrinsics. Raw ``K`` is also accepted at
        this numerical solver boundary for synthetic/reference tests.

    config:
        SfM quality thresholds. Uses ``SfMConfig()`` when omitted.

    Returns
    -------
    PnPResult
        Accepted pose and quality information, or an explicit rejection reason.
    """

    if config is None:
        config = SfMConfig()

    n_correspondences = len(points3d)

    if n_correspondences < config.min_pnp_correspondences:
        return PnPResult(
            accepted=False,
            pose=None,
            inlier_indices=np.empty(0, dtype=int),
            reprojection_rms_px=None,
            inlier_ratio=0.0,
            reason=FailureReason.INSUFFICIENT_PNP_CORRESPONDENCES,
        )

    if isinstance(intrinsics, CameraIntrinsics):
        K = intrinsics.K
        distortion = intrinsics.distortion
    else:
        K = np.asarray(intrinsics, dtype=float)
        distortion = None

    points3d = np.ascontiguousarray(
        points3d,
        dtype=np.float64,
    )
    points2d = np.ascontiguousarray(
        points2d,
        dtype=np.float64,
    )

    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d,
        points2d,
        K,
        distortion,
        iterationsCount=config.pnp_iterations,
        reprojectionError=config.pnp_reprojection_error_px,
        confidence=config.pnp_confidence,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if (
        not success
        or inliers is None
        or len(inliers) == 0
    ):
        return PnPResult(
            accepted=False,
            pose=None,
            inlier_indices=np.empty(0, dtype=int),
            reprojection_rms_px=None,
            inlier_ratio=0.0,
            reason=FailureReason.PNP_FAILED,
        )

    inlier_indices = inliers.ravel().astype(int)
    n_inliers = len(inlier_indices)

    inlier_ratio = float(
        n_inliers / n_correspondences
    )

    if n_inliers < config.min_pnp_inliers:
        return PnPResult(
            accepted=False,
            pose=None,
            inlier_indices=inlier_indices,
            reprojection_rms_px=None,
            inlier_ratio=inlier_ratio,
            reason=FailureReason.INSUFFICIENT_PNP_INLIERS,
        )

    if inlier_ratio < config.min_pnp_inlier_ratio:
        return PnPResult(
            accepted=False,
            pose=None,
            inlier_indices=inlier_indices,
            reprojection_rms_px=None,
            inlier_ratio=inlier_ratio,
            reason=FailureReason.LOW_PNP_INLIER_RATIO,
        )

    # Refine only the RANSAC-supported correspondences. If LM refinement
    # fails numerically, retain the valid RANSAC pose instead of discarding it.
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            points3d[inlier_indices],
            points2d[inlier_indices],
            K,
            distortion,
            rvec,
            tvec,
        )
    except cv2.error:
        pass

    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()

    projected = reproject(
        K,
        R,
        t,
        points3d[inlier_indices],
    )

    squared_errors = np.sum(
        (
            projected
            - points2d[inlier_indices]
        )
        ** 2,
        axis=1,
    )

    reprojection_rms_px = float(
        np.sqrt(np.mean(squared_errors))
    )

    if (
        reprojection_rms_px
        > config.max_pnp_rms_px
    ):
        return PnPResult(
            accepted=False,
            pose=None,
            inlier_indices=inlier_indices,
            reprojection_rms_px=reprojection_rms_px,
            inlier_ratio=inlier_ratio,
            reason=FailureReason.HIGH_PNP_REPROJECTION_ERROR,
        )

    return PnPResult(
        accepted=True,
        pose=CameraPose(
            R=R,
            t=t,
        ),
        inlier_indices=inlier_indices,
        reprojection_rms_px=reprojection_rms_px,
        inlier_ratio=inlier_ratio,
        reason=None,
    )
