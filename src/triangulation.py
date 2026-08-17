"""DLT triangulation and reprojection utilities."""

import numpy as np


def triangulate_two_view(
    P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray
) -> np.ndarray:
    """Vectorized two-view DLT. P are 3x4 projections; x are (N,2) in the same
    coordinate system as P (normalized or pixel). Returns (N,3)."""
    n = len(x1)
    A = np.empty((n, 4, 4))
    A[:, 0] = x1[:, 0:1] * P1[2] - P1[0]
    A[:, 1] = x1[:, 1:2] * P1[2] - P1[1]
    A[:, 2] = x2[:, 0:1] * P2[2] - P2[0]
    A[:, 3] = x2[:, 1:2] * P2[2] - P2[1]
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1]
    return X[:, :3] / X[:, 3:4]


def triangulate_multi_view(
    P_list: list[np.ndarray], x_list: list[np.ndarray]
) -> np.ndarray:
    """Multi-view DLT for a single point. x_list entries are (2,)."""
    rows = []
    for P, x in zip(P_list, x_list):
        rows.append(x[0] * P[2] - P[0])
        rows.append(x[1] * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def reproject(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Project world points (N,3) to pixels (N,2)."""
    Xc = X @ R.T + t
    xn = Xc[:, :2] / Xc[:, 2:3]
    return xn @ K[:2, :2].T + K[:2, 2]


def point_depths(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    return (X @ R.T + t)[:, 2]


def triangulation_angles_deg(
    C1: np.ndarray, C2: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Angle at each 3D point subtended by camera centers C1, C2."""
    r1 = C1 - X
    r2 = C2 - X
    cos = np.sum(r1 * r2, axis=1) / (
        np.linalg.norm(r1, axis=1) * np.linalg.norm(r2, axis=1) + 1e-15
    )
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))
