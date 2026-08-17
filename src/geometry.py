"""Two-view geometry from scratch: essential matrix, RANSAC, pose recovery."""

import numpy as np

rng = np.random.default_rng(0)


def to_normalized(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixel coords (N,2) -> normalized camera coords (N,2) via K^-1."""
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    xn = (np.linalg.inv(K) @ ph.T).T
    return xn[:, :2] / xn[:, 2:3]


def hartley_normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translate to centroid, scale mean distance to sqrt(2). Returns (pts_n, T)."""
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    ph = np.hstack([pts, np.ones((len(pts), 1))])
    return (T @ ph.T).T[:, :2], T


def estimate_essential_8pt(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """8-point algorithm on normalized-camera coords with Hartley conditioning.

    x2^T E x1 = 0. Enforces the (1,1,0) singular-value constraint.
    """
    x1n, T1 = hartley_normalize(x1)
    x2n, T2 = hartley_normalize(x2)
    u1, v1 = x1n[:, 0], x1n[:, 1]
    u2, v2 = x2n[:, 0], x2n[:, 1]
    A = np.column_stack(
        [u2 * u1, u2 * v1, u2, v2 * u1, v2 * v1, v2, u1, v1, np.ones_like(u1)]
    )
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    E = T2.T @ E @ T1  # undo conditioning
    U, _, Vt = np.linalg.svd(E)
    E = U @ np.diag([1.0, 1.0, 0.0]) @ Vt
    return E / np.linalg.norm(E)


def sampson_distance(E: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """First-order geometric error of x2^T E x1 = 0 (normalized coords)."""
    x1h = np.hstack([x1, np.ones((len(x1), 1))])
    x2h = np.hstack([x2, np.ones((len(x2), 1))])
    Ex1 = x1h @ E.T  # rows: E @ x1_i
    Etx2 = x2h @ E  # rows: E^T @ x2_i
    num = np.sum(x2h * Ex1, axis=1) ** 2
    den = Ex1[:, 0] ** 2 + Ex1[:, 1] ** 2 + Etx2[:, 0] ** 2 + Etx2[:, 1] ** 2
    return num / np.maximum(den, 1e-15)


def ransac_essential(
    x1: np.ndarray,
    x2: np.ndarray,
    thresh: float = 1e-5,
    iters: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC over 8-point samples; refit on inliers. Coords must be normalized.

    thresh is on squared Sampson distance in normalized coords
    (1e-5 ~ (2-3 px / f)^2 for f~1500).
    """
    n = len(x1)
    best_inliers = np.zeros(n, dtype=bool)
    for _ in range(iters):
        idx = rng.choice(n, 8, replace=False)
        try:
            E = estimate_essential_8pt(x1[idx], x2[idx])
        except np.linalg.LinAlgError:
            continue
        inliers = sampson_distance(E, x1, x2) < thresh
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    if best_inliers.sum() < 8:
        raise RuntimeError("RANSAC failed to find enough inliers")
    E = estimate_essential_8pt(x1[best_inliers], x2[best_inliers])
    best_inliers = sampson_distance(E, x1, x2) < thresh
    return E, best_inliers


def decompose_essential(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Four (R, t) candidates from E via the W-matrix construction."""
    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def choose_pose(
    candidates: list[tuple[np.ndarray, np.ndarray]],
    x1: np.ndarray,
    x2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cheirality check: pick the (R, t) putting the most points in front of
    both cameras. Camera 1 is [I|0]; coords normalized. Returns (R, t, mask)."""
    from .triangulation import triangulate_two_view

    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    best = None
    for R, t in candidates:
        P2 = np.hstack([R, t.reshape(3, 1)])
        X = triangulate_two_view(P1, P2, x1, x2)
        z1 = X[:, 2]
        z2 = (X @ R.T + t)[:, 2]
        mask = (z1 > 0) & (z2 > 0)
        if best is None or mask.sum() > best[2].sum():
            best = (R, t, mask)
    return best
