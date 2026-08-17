"""DLT triangulation, geometric validation, and reprojection utilities."""

from __future__ import annotations

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
from .validation.diagnostics import FailureReason


@dataclass
class TriangulationResult:
    """Explicit result of triangulating a multi-view feature track."""
    accepted: bool
    xyz: np.ndarray | None = None
    parallax_deg: float | None = None
    max_reprojection_error_px: float | None = None
    reason: FailureReason | str | None = None


def triangulate_two_view(
    P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """Vectorized two-view DLT. P are 3x4 projections; x are (N,2) in normalized coords.
    Returns (N,3) points in world frame."""
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    if x1.ndim == 1:
        x1 = x1.reshape(1, 2)
    if x2.ndim == 1:
        x2 = x2.reshape(1, 2)
    n = len(x1)
    A = np.empty((n, 4, 4))
    A[:, 0] = x1[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = x1[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = x2[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = x2[:, 1:2] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1]
    w = X[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return X[:, :3] / w


def triangulate_multi_view(
    P_list: list[np.ndarray], x_list: list[np.ndarray]
) -> np.ndarray:
    """Multi-view DLT for a single point from >=2 views. x_list entries are (2,)."""
    rows = []
    for P, x in zip(P_list, x_list):
        rows.append(x[0] * P[2] - P[0])
        rows.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    w = X[3] if abs(X[3]) > 1e-12 else 1e-12
    return X[:3] / w


def triangulate_track(
    obs: dict[int, int],
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig | None = None,
) -> TriangulationResult:
    """Triangulate a single track observing multiple registered cameras with full validation."""
    if config is None:
        config = SfMConfig()

    views = [img for img in obs if img in poses]
    if len(views) < 2:
        return TriangulationResult(
            accepted=False,
            reason=FailureReason.INSUFFICIENT_MATCHES,
        )

    P_list = []
    x_list = []
    for img in views:
        R, t = poses[img]
        P_list.append(np.hstack([R, t.reshape(3, 1)]))
        pt_norm = to_normalized(keypoints[img][obs[img]].reshape(1, 2), K)[0]
        x_list.append(pt_norm)

    X = triangulate_multi_view(P_list, x_list)

    if not np.all(np.isfinite(X)):
        return TriangulationResult(
            accepted=False,
            reason=FailureReason.ESSENTIAL_ESTIMATION_FAILED,
        )

    # 1. Cheirality / positive depth in all observing views
    for img in views:
        R, t = poses[img]
        depth = point_depths(R, t, X.reshape(1, 3))[0]
        if depth <= 0:
            return TriangulationResult(
                accepted=False,
                xyz=X,
                reason=FailureReason.NEGATIVE_DEPTH,
            )

    # 2. Triangulation parallax angle
    C1 = camera_center(*poses[views[0]])
    C2 = camera_center(*poses[views[-1]])
    parallax = float(triangulation_angles_deg(C1, C2, X.reshape(1, 3))[0])
    if parallax < config.min_parallax_deg:
        return TriangulationResult(
            accepted=False,
            xyz=X,
            parallax_deg=parallax,
            reason=FailureReason.LOW_PARALLAX,
        )

    # 3. Maximum reprojection error across all registered views
    max_err = 0.0
    for img in views:
        R, t = poses[img]
        proj = reproject(K, R, t, X.reshape(1, 3))[0]
        err = float(np.linalg.norm(proj - keypoints[img][obs[img]]))
        if err > max_err:
            max_err = err
        if err > config.max_triangulation_error_px:
            return TriangulationResult(
                accepted=False,
                xyz=X,
                parallax_deg=parallax,
                max_reprojection_error_px=max_err,
                reason=FailureReason.HIGH_REPROJECTION_ERROR,
            )

    return TriangulationResult(
        accepted=True,
        xyz=X,
        parallax_deg=parallax,
        max_reprojection_error_px=max_err,
        reason=None,
    )
