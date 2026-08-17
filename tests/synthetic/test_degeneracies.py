"""Tests for deliberate geometric degeneracies and failure detection."""

import numpy as np
from src.bundle_adjustment import rodrigues_to_R
from src.config import SfMConfig
from src.geometry.projection import reproject
from src.initialization import evaluate_pair
from src.pnp import estimate_camera_pose
from src.triangulation import triangulate_track
from src.validation.degeneracy import (
    check_cheirality,
    check_parallax,
    check_planar_degeneracy,
    check_pure_rotation,
)
from src.validation.diagnostics import FailureReason

rng = np.random.default_rng(123)


def test_pure_rotation_and_zero_baseline():
    """Pure rotation produces near-zero baseline and unresolvable depth scale."""
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]])
    R1, t1 = np.eye(3), np.zeros(3)
    R2 = rodrigues_to_R(np.array([0.05, 0.0, 0.0]))
    t2 = np.zeros(3)  # Exact pure rotation

    assert check_pure_rotation(t2) is True

    # 3D points
    X = rng.uniform([-1, -1, 4], [1, 1, 8], size=(50, 3))
    x1 = reproject(K, R1, t1, X)
    x2 = reproject(K, R2, t2, X)
    matches = np.column_stack([np.arange(len(X)), np.arange(len(X))])
    keypoints = [x1, x2]

    # Evaluate pair initialization
    ev = evaluate_pair(0, 1, matches, keypoints, K, SfMConfig(min_init_inliers=30))
    # Must be rejected due to geometric degeneracy (low parallax or poor cheirality or essential failure)
    assert ev.accepted is False
    assert ev.rejection_reason in (
        FailureReason.LOW_PARALLAX,
        FailureReason.POOR_CHEIRALITY,
        FailureReason.ESSENTIAL_ESTIMATION_FAILED,
    )


def test_planar_scene_degeneracy():
    """Planar points have zero variance along normal direction."""
    # Points lying strictly on plane z = 5
    xy = rng.uniform([-2, -2], [2, 2], size=(100, 2))
    z = np.full((100, 1), 5.0)
    X_planar = np.hstack([xy, z])

    is_planar, var_ratio = check_planar_degeneracy(X_planar)
    assert is_planar is True
    assert var_ratio < 1e-6

    # General 3D points are non-planar
    X_3d = rng.uniform([-2, -2, 3], [2, 2, 8], size=(100, 3))
    is_planar_3d, var_ratio_3d = check_planar_degeneracy(X_3d)
    assert is_planar_3d is False
    assert var_ratio_3d > 1e-2


def test_points_behind_camera():
    """Points with negative depth fail cheirality and triangulation."""
    R = np.eye(3)
    t = np.zeros(3)
    X_behind = np.array([[0.0, 0.0, -3.0], [1.0, 1.0, -5.0]])

    passed, ratio = check_cheirality(R, t, X_behind)
    assert passed is False
    assert ratio == 0.0

    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]])
    poses = {0: (R, t), 1: (R, np.array([-1.0, 0, 0]))}
    kp0 = reproject(K, poses[0][0], poses[0][1], X_behind[:1])
    kp1 = reproject(K, poses[1][0], poses[1][1], X_behind[:1])
    res = triangulate_track({0: 0, 1: 0}, poses, [kp0, kp1], K, SfMConfig())
    # Fails cheirality / depth
    assert res.accepted is False
    assert res.reason == FailureReason.NEGATIVE_DEPTH


def test_pnp_with_gross_outliers():
    """OpenCV PnP rejects or handles high-noise correspondences."""
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]])
    R_gt = rodrigues_to_R(np.array([0.1, -0.05, 0.02]))
    t_gt = np.array([0.5, -0.2, 0.1])
    X = rng.uniform([-1, -1, 3], [1, 1, 7], size=(60, 3))
    x = reproject(K, R_gt, t_gt, X)

    # Corrupt 40% with random noise
    n_corrupt = 24
    x[:n_corrupt] = rng.uniform([0, 0], [1000, 1000], size=(n_corrupt, 2))

    res = estimate_camera_pose(X, x, K, SfMConfig(min_pnp_inliers=20))
    assert res.accepted is True
    assert res.pose is not None
    assert np.allclose(res.pose.R, R_gt, atol=1e-3)
    assert np.allclose(res.pose.t, t_gt, atol=1e-3)
    # The corrupted points should be filtered out
    assert not any(idx < n_corrupt for idx in res.inlier_indices)


if __name__ == "__main__":
    test_pure_rotation_and_zero_baseline()
    test_planar_scene_degeneracy()
    test_points_behind_camera()
    test_pnp_with_gross_outliers()
    print("test_degeneracies: OK")
