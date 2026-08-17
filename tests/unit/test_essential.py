"""Unit tests for essential matrix computation, decomposition, and pose selection."""

import numpy as np
from src.geometry.essential import (
    choose_pose,
    decompose_essential,
    estimate_essential_8pt,
    hartley_normalize,
    ransac_essential,
    sampson_distance,
)
from src.geometry.projection import reproject


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
    assert np.allclose(R_sel, R, atol=1e-5)
    assert np.allclose(t_sel / np.linalg.norm(t_sel), t / np.linalg.norm(t), atol=1e-5)


if __name__ == "__main__":
    test_hartley_normalize()
    test_essential_matrix_properties()
    print("test_essential: OK")
