"""Camera projection and image coordinate transformation utilities."""

import numpy as np


def to_normalized(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel coords (N,2) -> normalized camera coords (N,2) via K^-1."""
    pts = np.asarray(pts, dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    xn = (np.linalg.inv(K) @ ph.T).T
    return xn[:, :2] / xn[:, 2:3]


def reproject(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project 3D world points (N,3) to 2D pixel coordinates (N,2).

    X_cam = R @ X + t
    x_norm = X_cam[:2] / X_cam[2]
    x_pixel = K @ [x_norm, 1]
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, 3)
    Xc = X @ R.T + t.reshape(1, 3)
    xn = Xc[:, :2] / Xc[:, 2:3]
    return xn @ K[:2, :2].T + K[:2, 2]


def point_depths(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Compute z-depth of 3D points (N,3) in camera coordinates."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, 3)
    return (X @ R.T + t.reshape(1, 3))[:, 2]


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Optical center C in world frame satisfying R @ C + t = 0 => C = -R.T @ t."""
    return -R.T @ t.reshape(3)


def triangulation_angles_deg(
    C1: np.ndarray, C2: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Parallax angle in degrees subtended by camera centers C1, C2 at points X (N,3)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, 3)
    C1 = np.asarray(C1, dtype=float).reshape(1, 3)
    C2 = np.asarray(C2, dtype=float).reshape(1, 3)
    r1 = C1 - X
    r2 = C2 - X
    norm1 = np.linalg.norm(r1, axis=1)
    norm2 = np.linalg.norm(r2, axis=1)
    cos = np.sum(r1 * r2, axis=1) / (norm1 * norm2 + 1e-15)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
