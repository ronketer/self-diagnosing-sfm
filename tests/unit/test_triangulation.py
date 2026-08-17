"""Unit tests for DLT triangulation and validation checks."""

import numpy as np
from src.config import SfMConfig
from src.geometry.projection import reproject
from src.triangulation import (
    TriangulationResult,
    triangulate_multi_view,
    triangulate_track,
    triangulate_two_view,
)
from src.validation.diagnostics import FailureReason


def test_two_view_and_multi_view_dlt():
    # 3 cameras along X axis
    poses = {
        0: (np.eye(3), np.zeros(3)),
        1: (np.eye(3), np.array([-1.0, 0.0, 0.0])),
        2: (np.eye(3), np.array([1.0, 0.0, 0.0])),
    }
    X_gt = np.array([0.5, -0.3, 4.0])

    P_list = [np.hstack([R, t.reshape(3, 1)]) for R, t in poses.values()]
    x_list = [reproject(np.eye(3), R, t, X_gt.reshape(1, 3))[0] for R, t in poses.values()]

    # Two-view DLT
    X_2v = triangulate_two_view(P_list[0], P_list[1], x_list[0].reshape(1, 2), x_list[1].reshape(1, 2))[0]
    assert np.allclose(X_2v, X_gt, atol=1e-8)

    # Multi-view DLT
    X_mv = triangulate_multi_view(P_list, x_list)
    assert np.allclose(X_mv, X_gt, atol=1e-8)


def test_triangulate_track_validation():
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]], dtype=float)
    poses = {
        0: (np.eye(3), np.zeros(3)),
        1: (np.eye(3), np.array([-1.0, 0.0, 0.0])),
    }
    X_front = np.array([[0.0, 0.0, 5.0]])
    X_back = np.array([[0.0, 0.0, -5.0]])  # Behind camera

    kp0_f = reproject(K, poses[0][0], poses[0][1], X_front)[0]
    kp1_f = reproject(K, poses[1][0], poses[1][1], X_front)[0]

    keypoints = [np.array([kp0_f]), np.array([kp1_f])]
    obs = {0: 0, 1: 0}

    # Valid track
    res = triangulate_track(obs, poses, keypoints, K, SfMConfig())
    assert res.accepted
    assert res.xyz is not None
    assert np.allclose(res.xyz, X_front[0], atol=1e-6)

    # Point behind camera -> negative depth rejection
    kp0_b = reproject(K, poses[0][0], poses[0][1], X_back)[0]
    kp1_b = reproject(K, poses[1][0], poses[1][1], X_back)[0]
    keypoints_back = [np.array([kp0_b]), np.array([kp1_b])]
    res_back = triangulate_track(obs, poses, keypoints_back, K, SfMConfig())
    assert res_back.accepted is False
    assert res_back.reason == FailureReason.NEGATIVE_DEPTH


def test_multiview_triangulation_best_parallax_pair():
    """Regression test: parallax must be computed across ALL unique camera pairs.

    Camera 0 and Camera 2 have almost identical centers (tiny baseline, parallax << 1.0 deg).
    Camera 0 and Camera 1 have wide baseline (parallax > 10.0 deg).
    Arrange observation order {0: kp0, 2: kp2, 1: kp1} where first/last pair (0, 2) has tiny parallax.
    """
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]], dtype=float)
    X_gt = np.array([[0.0, 0.0, 5.0]])

    # Camera 0 at origin
    # Camera 1 at x = 1.0 (baseline 1.0m, parallax ≈ 11.3 deg)
    # Camera 2 at x = 0.001 (baseline 0.001m, parallax ≈ 0.011 deg)
    poses = {
        0: (np.eye(3), np.zeros(3)),
        1: (np.eye(3), np.array([-1.0, 0.0, 0.0])),
        2: (np.eye(3), np.array([-0.001, 0.0, 0.0])),
    }

    kp0 = reproject(K, poses[0][0], poses[0][1], X_gt)[0]
    kp1 = reproject(K, poses[1][0], poses[1][1], X_gt)[0]
    kp2 = reproject(K, poses[2][0], poses[2][1], X_gt)[0]

    keypoints = [np.array([kp0]), np.array([kp1]), np.array([kp2])]

    # Dict order: 0, then 2, then 1 (first/last is (0, 2) if using [0] and [-1])
    obs = {0: 0, 2: 0, 1: 0}

    cfg = SfMConfig(min_parallax_deg=1.0)
    res = triangulate_track(obs, poses, keypoints, K, cfg)

    assert res.accepted is True, f"Triangulation rejected unexpectedly: reason={res.reason}"
    assert res.parallax_deg is not None and res.parallax_deg > 5.0
    assert res.best_parallax_pair in ((0, 1), (1, 0), (1, 2), (2, 1))
    assert np.allclose(res.xyz, X_gt[0], atol=1e-3)


if __name__ == "__main__":
    test_two_view_and_multi_view_dlt()
    test_triangulate_track_validation()
    test_multiview_triangulation_best_parallax_pair()
    print("test_triangulation.py: ALL PASSED")
