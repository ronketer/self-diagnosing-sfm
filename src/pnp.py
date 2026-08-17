"""Perspective-n-Point from scratch: DLT + RANSAC + nonlinear refinement."""

import numpy as np
from scipy.optimize import least_squares

from .triangulation import reproject

rng = np.random.default_rng(1)


def dlt_pnp(pts3d: np.ndarray, pts2d: np.ndarray, K: np.ndarray):
    """Linear PnP: solve P = K[R|t] from >=6 2D-3D pairs, then extract (R, t)
    by orthogonalizing the rotation block. Returns (R, t) or None."""
    n = len(pts3d)
    if n < 6:
        return None
    # Work in normalized image coords so the solved M ~ [R|t]
    xn = (np.linalg.inv(K) @ np.hstack([pts2d, np.ones((n, 1))]).T).T
    # Condition the 3D points (centroid at origin, unit mean norm) — plain
    # DLT is numerically fragile without this
    c3 = pts3d.mean(axis=0)
    s3 = np.linalg.norm(pts3d - c3, axis=1).mean()
    s3 = max(s3, 1e-12)
    Xc = (pts3d - c3) / s3
    A = np.zeros((2 * n, 12))
    Xh = np.hstack([Xc, np.ones((n, 1))])
    A[0::2, 0:4] = Xh
    A[0::2, 8:12] = -xn[:, 0:1] * Xh
    A[1::2, 4:8] = Xh
    A[1::2, 8:12] = -xn[:, 1:2] * Xh
    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4)

    # Fix sign so points end up in front of the camera
    if np.median(Xh @ M[2]) < 0:
        M = -M

    # Undo 3D conditioning: M_orig = M @ [[I/s, -c/s], [0, 1]]
    M = np.hstack([M[:, :3] / s3, (M[:, 3] - M[:, :3] @ c3 / s3).reshape(3, 1)])

    R_raw, t_raw = M[:, :3], M[:, 3]
    U, S, Vt2 = np.linalg.svd(R_raw)
    R = U @ Vt2
    if np.linalg.det(R) < 0:
        R = -R
        t_raw = -t_raw
    t = t_raw / S.mean()
    return R, t


def ransac_pnp(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    thresh_px: float = 4.0,
    iters: int = 3000,
):
    """LO-RANSAC: 6-point DLT hypotheses, then iterated nonlinear
    refine-on-inliers + recount. Returns (R, t, inlier_mask) or None."""
    n = len(pts3d)
    if n < 6:
        return None
    best_mask = np.zeros(n, dtype=bool)
    best_pose = None
    for _ in range(iters):
        idx = rng.choice(n, 6, replace=False)
        pose = dlt_pnp(pts3d[idx], pts2d[idx], K)
        if pose is None:
            continue
        R, t = pose
        err = np.linalg.norm(reproject(K, R, t, pts3d) - pts2d, axis=1)
        mask = err < thresh_px
        if mask.sum() > best_mask.sum():
            best_mask, best_pose = mask, pose
    if best_pose is None or best_mask.sum() < 6:
        return None
    # local optimization: refine on inliers, recount, repeat while improving
    R, t = best_pose
    for _ in range(4):
        R, t = refine_pose(R, t, pts3d[best_mask], pts2d[best_mask], K)
        err = np.linalg.norm(reproject(K, R, t, pts3d) - pts2d, axis=1)
        mask = err < thresh_px
        if mask.sum() <= best_mask.sum() and (mask == best_mask).all():
            break
        if mask.sum() >= best_mask.sum():
            best_mask = mask
    return R, t, best_mask


def refine_pose(R, t, pts3d, pts2d, K):
    """Motion-only refinement over [rvec, t] with scipy least_squares."""
    from .bundle_adjustment import rodrigues_to_R, R_to_rodrigues

    x0 = np.concatenate([R_to_rodrigues(R), t])

    def resid(p):
        Rp = rodrigues_to_R(p[:3])
        return (reproject(K, Rp, p[3:], pts3d) - pts2d).ravel()

    sol = least_squares(resid, x0, method="lm", max_nfev=200)
    return rodrigues_to_R(sol.x[:3]), sol.x[3:]
