"""Degeneracy detection and geometric sanity checks for Structure-from-Motion."""

from __future__ import annotations

import cv2
import numpy as np

from ..geometry.projection import triangulation_angles_deg


def check_ground_truth_baseline(t: np.ndarray, min_baseline: float = 1e-4) -> bool:
    """Test if a known ground-truth physical translation is negligible.

    NOTE: This is strictly for synthetic/ground-truth checks. Essential-matrix
    decomposition yields a unit-norm direction (||t|| == 1), so physical pure
    rotation in production SfM MUST be detected via triangulation parallax angle.
    """
    return float(np.linalg.norm(t)) < min_baseline


# Alias for backwards compatibility in synthetic tests; documented as synthetic-only
check_pure_rotation = check_ground_truth_baseline


def evaluate_homography_support(
    pts_px1: np.ndarray,
    pts_px2: np.ndarray,
    thresh_px: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray, int, float]:
    """Fit a projective Homography using OpenCV RANSAC to evaluate planar/rotational support.

    Returns (H, inlier_mask, n_inliers, inlier_ratio).
    """
    pts1 = np.ascontiguousarray(pts_px1, dtype=np.float32)
    pts2 = np.ascontiguousarray(pts_px2, dtype=np.float32)

    if len(pts1) < 4:
        return None, np.zeros(len(pts1), dtype=bool), 0, 0.0

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, thresh_px)
    if H is None or mask is None:
        return None, np.zeros(len(pts1), dtype=bool), 0, 0.0

    inlier_mask = mask.ravel().astype(bool)
    n_inliers = int(inlier_mask.sum())
    inlier_ratio = float(n_inliers / max(len(pts1), 1))
    return H, inlier_mask, n_inliers, inlier_ratio


def check_planar_degeneracy(
    points3d: np.ndarray, max_variance_ratio: float = 1e-3
) -> tuple[bool, float]:
    """Secondary diagnostic: check if reconstructed 3D points lie on a plane using PCA variance ratio.

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
