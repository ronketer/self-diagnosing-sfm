"""Pose transformations, rotation metrics, and Sim(3) Umeyama alignment."""

import numpy as np


def rotation_error_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic rotation angle error in degrees between two rotation matrices."""
    cos = np.clip((np.trace(R1 @ R2.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform (s, R, t) minimizing ||s * R @ src.T + t - dst.T||^2."""
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (sc**2).sum() / len(src)
    s = np.trace(np.diag(D) @ S) / max(var_s, 1e-15)
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def relative_pose(
    R1: np.ndarray, t1: np.ndarray, R2: np.ndarray, t2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute relative pose from camera 1 to camera 2.

    X_cam1 = R1 @ X_w + t1
    X_cam2 = R2 @ X_w + t2 = R2 @ R1.T @ (X_cam1 - t1) + t2
           = (R2 @ R1.T) @ X_cam1 + (t2 - R2 @ R1.T @ t1)
    """
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return R_rel, t_rel
