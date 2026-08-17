"""Degeneracy detection and geometric sanity checks for Structure-from-Motion."""

import numpy as np
from ..geometry.projection import triangulation_angles_deg


def check_pure_rotation(t: np.ndarray, min_baseline: float = 1e-4) -> bool:
    """Returns True if the translation is negligible (pure rotation degeneracy)."""
    return float(np.linalg.norm(t)) < min_baseline


def check_planar_degeneracy(
    points3d: np.ndarray, max_variance_ratio: float = 1e-3
) -> tuple[bool, float]:
    """Check if 3D points lie on a plane using PCA / SVD variance ratio.

    Returns (is_planar, min_var_ratio).
    """
    pts = np.asarray(points3d, dtype=float)
    if len(pts) < 4:
        return True, 0.0
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    _, s, _ = np.linalg.svd(shifted)
    variances = s ** 2
    var_ratio = float(variances[-1] / (variances.sum() + 1e-15))
    return var_ratio < max_variance_ratio, var_ratio


def check_parallax(
    C1: np.ndarray, C2: np.ndarray, points3d: np.ndarray, min_parallax_deg: float = 1.0
) -> tuple[bool, float, float]:
    """Check if triangulation parallax is sufficient.

    Returns (is_sufficient, median_deg, p90_deg).
    """
    if len(points3d) == 0:
        return False, 0.0, 0.0
    angles = triangulation_angles_deg(C1, C2, points3d)
    med = float(np.median(angles))
    p90 = float(np.percentile(angles, 90))
    return med >= min_parallax_deg, med, p90


def check_cheirality(
    R: np.ndarray, t: np.ndarray, points3d: np.ndarray, min_ratio: float = 0.75
) -> tuple[bool, float]:
    """Check if points have positive depth in both camera 1 [I|0] and camera 2 [R|t].

    Returns (passes_cheirality, positive_ratio).
    """
    pts = np.asarray(points3d, dtype=float)
    if len(pts) == 0:
        return False, 0.0
    z1 = pts[:, 2]
    z2 = (pts @ R.T + t.reshape(1, 3))[:, 2]
    valid = (z1 > 0) & (z2 > 0)
    ratio = float(np.mean(valid))
    return ratio >= min_ratio, ratio
