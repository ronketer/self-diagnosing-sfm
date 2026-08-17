"""Unit tests for projection geometry functions."""

import numpy as np
from src.geometry.projection import (
    camera_center,
    point_depths,
    reproject,
    to_normalized,
    triangulation_angles_deg,
)


def test_to_normalized_and_reproject():
    K = np.array([[1000.0, 0.0, 500.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    R = np.eye(3)
    t = np.array([0.0, 0.0, 2.0])
    X = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 8.0]])

    # Reproject to pixel coordinates
    # X_cam for point 0: [0, 0, 7] -> u = 1000*0/7 + 500 = 500, v = 1000*0/7 + 400 = 400
    # X_cam for point 1: [1, 2, 10] -> u = 1000*1/10 + 500 = 600, v = 1000*2/10 + 400 = 600
    pts_px = reproject(K, R, t, X)
    expected_px = np.array([[500.0, 400.0], [600.0, 600.0]])
    assert np.allclose(pts_px, expected_px, atol=1e-6)

    # Invert via to_normalized
    pts_norm = to_normalized(pts_px, K)
    expected_norm = np.array([[0.0, 0.0], [0.1, 0.2]])
    assert np.allclose(pts_norm, expected_norm, atol=1e-6)


def test_camera_center():
    # R: 90 deg rotation about Z
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    t = np.array([2.0, 3.0, 1.0])
    # C = -R.T @ t = - [[0, 1, 0], [-1, 0, 0], [0, 0, 1]] @ [2, 3, 1] = - [3, -2, 1] = [-3, 2, -1]
    C = camera_center(R, t)
    assert np.allclose(C, np.array([-3.0, 2.0, -1.0]), atol=1e-9)
    assert np.allclose(R @ C + t, np.zeros(3), atol=1e-9)


def test_point_depths():
    R = np.eye(3)
    t = np.array([0.0, 0.0, 1.5])
    X = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -2.0]])
    depths = point_depths(R, t, X)
    assert np.allclose(depths, [4.5, -0.5], atol=1e-9)


def test_triangulation_angles():
    C1 = np.array([0.0, 0.0, 0.0])
    C2 = np.array([2.0, 0.0, 0.0])
    X = np.array([[1.0, 0.0, 1.0]])  # Right angle triangle, 45 + 45 = 90 deg
    angle = triangulation_angles_deg(C1, C2, X)[0]
    assert np.isclose(angle, 90.0, atol=1e-4)


if __name__ == "__main__":
    test_to_normalized_and_reproject()
    test_camera_center()
    test_point_depths()
    test_triangulation_angles()
    print("test_projection: OK")
