"""Sparse bundle adjustment from scratch on top of scipy.optimize.least_squares."""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> rotation matrix, closed-form Rodrigues formula."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def R_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle (3,)."""
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:  # near 180 deg: axis from R + I
        A = (R + np.eye(3)) / 2
        axis = np.sqrt(np.maximum(np.diag(A), 0))
        # fix signs using off-diagonals
        if axis[0] > 0:
            axis[1] *= np.sign(A[0, 1]) if A[0, 1] != 0 else 1
            axis[2] *= np.sign(A[0, 2]) if A[0, 2] != 0 else 1
        axis /= np.linalg.norm(axis)
        return theta * axis
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta / (2 * np.sin(theta)) * v


def _rotate(rvecs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation of pts[i] by rvecs[i]. Both (N,3)."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    cross = np.cross(k, pts)
    dot = np.sum(k * pts, axis=1, keepdims=True)
    return pts * cos + cross * sin + k * dot * (1 - cos)


def run_bundle_adjustment(
    K: np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    points3d: np.ndarray,
    observations: list[tuple[int, int, np.ndarray]],
    fixed_cams: set[int] | None = None,
    verbose: int = 0,
):
    """Minimize total reprojection error.

    observations: list of (img_idx, point3d_id, xy_pixel).
    fixed_cams: cameras excluded from the parameter vector (gauge fixing);
                defaults to the lowest-index camera.
    Returns (poses, points3d, rms_before, rms_after).
    """
    cam_ids = sorted(poses)
    if fixed_cams is None:
        fixed_cams = {cam_ids[0]}
    free_cams = [c for c in cam_ids if c not in fixed_cams]
    cam_slot = {c: i for i, c in enumerate(free_cams)}

    n_pts = len(points3d)
    obs_cam = np.array([o[0] for o in observations])
    obs_pt = np.array([o[1] for o in observations])
    obs_xy = np.array([o[2] for o in observations])

    x0 = np.concatenate(
        [
            np.concatenate([np.concatenate([R_to_rodrigues(poses[c][0]), poses[c][1]]) for c in free_cams])
            if free_cams
            else np.array([]),
            points3d.ravel(),
        ]
    )
    n_cam_params = 6 * len(free_cams)

    fixed_rvec = {c: R_to_rodrigues(poses[c][0]) for c in fixed_cams}
    fixed_t = {c: poses[c][1] for c in fixed_cams}

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(params):
        rvecs = np.empty((len(obs_cam), 3))
        ts = np.empty((len(obs_cam), 3))
        cams = params[:n_cam_params].reshape(-1, 6)
        for c in cam_ids:
            m = obs_cam == c
            if c in fixed_cams:
                rvecs[m], ts[m] = fixed_rvec[c], fixed_t[c]
            else:
                rvecs[m], ts[m] = cams[cam_slot[c], :3], cams[cam_slot[c], 3:]
        pts = params[n_cam_params:].reshape(-1, 3)[obs_pt]
        return rvecs, ts, pts

    def resid(params):
        rvecs, ts, pts = unpack(params)
        Xc = _rotate(rvecs, pts) + ts
        u = fx * Xc[:, 0] / Xc[:, 2] + cx
        v = fy * Xc[:, 1] / Xc[:, 2] + cy
        return np.concatenate([u - obs_xy[:, 0], v - obs_xy[:, 1]])

    # Jacobian sparsity: each residual depends on its camera's 6 params
    # and its point's 3 params.
    n_obs = len(observations)
    sparsity = lil_matrix((2 * n_obs, n_cam_params + 3 * n_pts), dtype=int)
    rows = np.arange(n_obs)
    for c in free_cams:
        m = np.where(obs_cam == c)[0]
        for k in range(6):
            sparsity[m, 6 * cam_slot[c] + k] = 1
            sparsity[n_obs + m, 6 * cam_slot[c] + k] = 1
    for k in range(3):
        sparsity[rows, n_cam_params + 3 * obs_pt + k] = 1
        sparsity[n_obs + rows, n_cam_params + 3 * obs_pt + k] = 1

    r0 = resid(x0)
    rms_before = np.sqrt(np.mean(r0[:n_obs] ** 2 + r0[n_obs:] ** 2))

    sol = least_squares(
        resid,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        x_scale="jac",
        ftol=1e-6,
        verbose=verbose,
    )

    r1 = sol.fun
    rms_after = np.sqrt(np.mean(r1[:n_obs] ** 2 + r1[n_obs:] ** 2))

    cams = sol.x[:n_cam_params].reshape(-1, 6)
    new_poses = dict(poses)
    for c in free_cams:
        new_poses[c] = (rodrigues_to_R(cams[cam_slot[c], :3]), cams[cam_slot[c], 3:])
    new_points = sol.x[n_cam_params:].reshape(-1, 3)
    return new_poses, new_points, rms_before, rms_after
