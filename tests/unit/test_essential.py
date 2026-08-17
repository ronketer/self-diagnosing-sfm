"""Unit tests for essential matrix computation, decomposition, and pose selection."""

import numpy as np
from src.config import SfMConfig
from src.geometry.essential import (
    choose_pose,
    decompose_essential,
    estimate_essential_8pt,
    hartley_normalize,
    ransac_essential,
    sampson_distance,
)
from src.geometry.projection import reproject
from src.initialization import evaluate_pair_geometry
from src.models import PairGeometry


def test_hartley_normalize():
    pts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    norm_pts, T = hartley_normalize(pts)
    assert np.allclose(norm_pts.mean(axis=0), [0.0, 0.0], atol=1e-9)
    d = np.linalg.norm(norm_pts, axis=1).mean()
    assert np.isclose(d, np.sqrt(2.0), atol=1e-9)


def test_essential_matrix_properties():
    # Construct exact essential matrix from known rotation and translation
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg around Z
    t = np.array([1.0, 0.0, 0.0])
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E_true = tx @ R
    E_true /= np.linalg.norm(E_true)

    # Decompose into 4 candidate poses
    candidates = decompose_essential(E_true)
    assert len(candidates) == 4

    # Generate synthetic 3D points in front of both cameras
    pts3d = np.array([
        [0.0, 0.0, 5.0],
        [0.5, 0.5, 6.0],
        [-0.5, 0.5, 4.5],
        [0.2, -0.4, 5.5],
        [0.3, 0.2, 7.0],
        [-0.2, -0.2, 6.5],
        [0.1, 0.4, 5.2],
        [-0.3, 0.1, 4.8],
    ])

    x1 = reproject(np.eye(3), np.eye(3), np.zeros(3), pts3d)
    x2 = reproject(np.eye(3), R, t, pts3d)

    # Check Sampson distance is ~0 for true E
    sd = sampson_distance(E_true, x1, x2)
    assert np.all(sd < 1e-12)

    # Estimate E with 8-point algorithm
    E_est = estimate_essential_8pt(x1, x2)
    assert np.allclose(np.abs(E_est), np.abs(E_true), atol=1e-4)

    # Cheirality pose selection
    R_sel, t_sel, mask = choose_pose(candidates, x1, x2)
    assert mask.all()
    assert np.allclose(R_sel, R, atol=1e-4)
    assert np.allclose(t_sel / np.linalg.norm(t_sel), t / np.linalg.norm(t), atol=1e-4)


def test_deterministic_essential_ransac():
    """Essential RANSAC must produce bitwise identical results for the same seed regardless of prior calls."""
    rng_local = np.random.default_rng(123)
    pts3d = rng_local.uniform([-1, -1, 3], [1, 1, 7], size=(50, 3))
    R = np.eye(3)
    t = np.array([0.5, 0.0, 0.0])

    x1 = reproject(np.eye(3), np.eye(3), np.zeros(3), pts3d)
    x2 = reproject(np.eye(3), R, t, pts3d)

    # Run standalone
    E1, inliers1 = ransac_essential(x1, x2, thresh=1e-5, iters=500, seed=42)

    # Run intermediate RANSAC calls to advance any arbitrary state
    for _ in range(5):
        dummy_pts = rng_local.uniform(-1, 1, size=(20, 2))
        try:
            ransac_essential(dummy_pts, dummy_pts, iters=50, seed=999)
        except RuntimeError:
            pass

    # Run again with same seed 42
    E2, inliers2 = ransac_essential(x1, x2, thresh=1e-5, iters=500, seed=42)

    assert np.allclose(E1, E2, atol=1e-12)
    assert np.array_equal(inliers1, inliers2)
    assert inliers1.sum() == inliers2.sum()


def test_pair_geometry_reuse_in_initialization():
    """Initialization must evaluate PairGeometry directly without re-running RANSAC."""
    pts3d = np.random.default_rng(7).uniform([-1, -1, 3], [1, 1, 7], size=(120, 3))
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]], dtype=float)
    R_gt = np.eye(3)
    t_gt = np.array([0.5, 0.0, 0.0])

    kp0 = reproject(K, np.eye(3), np.zeros(3), pts3d)
    kp1 = reproject(K, R_gt, t_gt, pts3d)

    x1 = (np.linalg.inv(K) @ np.hstack([kp0, np.ones((len(kp0), 1))]).T).T[:, :2]
    x2 = (np.linalg.inv(K) @ np.hstack([kp1, np.ones((len(kp1), 1))]).T).T[:, :2]

    E, inliers = ransac_essential(x1, x2, thresh=1e-5, iters=1000, seed=101)
    matches = np.column_stack([np.arange(len(kp0)), np.arange(len(kp0))])

    pg = PairGeometry(
        image_a=0,
        image_b=1,
        raw_matches=matches,
        verified_matches=matches[inliers],
        E=E,
        inlier_mask=inliers,
        n_raw_matches=len(matches),
        n_inliers=int(inliers.sum()),
        inlier_ratio=float(inliers.sum() / len(matches)),
    )

    eval_res = evaluate_pair_geometry(pg, [kp0, kp1], K, SfMConfig(min_init_inliers=50))
    assert eval_res.accepted is True
    assert eval_res.n_inliers == int(inliers.sum())
    assert np.allclose(eval_res.E, E)
    assert np.allclose(eval_res.R, R_gt, atol=1e-3)


if __name__ == "__main__":
    test_hartley_normalize()
    test_essential_matrix_properties()
    test_deterministic_essential_ransac()
    test_pair_geometry_reuse_in_initialization()
    print("test_essential.py: ALL PASSED")
