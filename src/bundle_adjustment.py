"""Sparse bundle adjustment with explicit camera-center baseline gauge constraints and OpenCV Rodrigues boundaries."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .config import SfMConfig
from .geometry.projection import camera_center
from .models import CameraIntrinsics, CameraPose


def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle (3,) -> 3x3 rotation matrix using OpenCV Rodrigues."""
    rvec = np.asarray(rvec, dtype=float).reshape(3, 1)
    return cv2.Rodrigues(rvec)[0]


def R_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle (3,) using OpenCV Rodrigues."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    return cv2.Rodrigues(R)[0].ravel()


def _rotate(rvecs: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues rotation of pts[i] by rvecs[i]. Both (N,3).

    Kept vectorized in pure NumPy for high performance during least_squares inner loop.
    """
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore"):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    cos, sin = np.cos(theta), np.sin(theta)
    cross = np.cross(k, pts)
    dot = np.sum(k * pts, axis=1, keepdims=True)
    return pts * cos + cross * sin + k * dot * (1.0 - cos)


def run_bundle_adjustment(
    intrinsics: CameraIntrinsics | np.ndarray,
    poses: dict[int, tuple[np.ndarray, np.ndarray] | CameraPose],
    points3d: np.ndarray,
    observations: list[tuple[int, int, np.ndarray]],
    anchor_camera: int | None = None,
    scale_camera: int | None = None,
    initial_baseline: float | None = None,
    fixed_cams: set[int] | None = None,
    config: SfMConfig | None = None,
    verbose: int = 0,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray, float, float]:
    """Jointly optimize camera poses and 3D landmark positions with explicit gauge fixing.

    Gauge Constraints:
    - Anchor camera (i0): excluded from the optimization vector, preserving its exact existing pose. In the normal initialization frame this pose is I, 0.
    - Scale camera (j0): scale penalty r_scale = lambda_s * (||C_j0 - C_i0|| - d0) preserves the initial baseline d0.

    Returns (optimized_poses, optimized_points3d, rms_before, rms_after).
    """
    if config is None:
        config = SfMConfig()

    if isinstance(intrinsics, CameraIntrinsics):
        K = intrinsics.K
    else:
        K = np.asarray(intrinsics, dtype=float)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # Normalize poses to tuple format
    poses_dict: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid, p in poses.items():
        if isinstance(p, CameraPose):
            poses_dict[cid] = (p.R, p.t)
        else:
            poses_dict[cid] = (p[0], p[1])

    cam_ids = sorted(poses_dict.keys())
    if anchor_camera is None:
        anchor_camera = cam_ids[0]

    if fixed_cams is None:
        fixed_cams = {anchor_camera}
    else:
        fixed_cams = set(fixed_cams) | {anchor_camera}

    free_cams = [c for c in cam_ids if c not in fixed_cams]
    cam_slot = {c: i for i, c in enumerate(free_cams)}

    n_pts = len(points3d)
    n_obs = len(observations)
    if n_obs == 0:
        return poses_dict, points3d, 0.0, 0.0

    obs_cam = np.array([o[0] for o in observations])
    obs_pt = np.array([o[1] for o in observations])
    obs_xy = np.array([o[2] for o in observations])

    # Convert rotations to rodrigues vectors at optimization boundary
    cam_params_list = []
    for c in free_cams:
        rvec = R_to_rodrigues(poses_dict[c][0])
        tvec = poses_dict[c][1]
        cam_params_list.extend([rvec, tvec])

    if free_cams:
        cam_params = np.concatenate(cam_params_list)
    else:
        cam_params = np.array([])

    x0 = np.concatenate([cam_params, points3d.ravel()])
    n_cam_params = 6 * len(free_cams)

    fixed_rvec = {c: R_to_rodrigues(poses_dict[c][0]) for c in fixed_cams}
    fixed_t = {c: poses_dict[c][1] for c in fixed_cams}

    # Determine baseline gauge d0
    if initial_baseline is None and scale_camera is not None and scale_camera in poses_dict:
        C_anchor = camera_center(poses_dict[anchor_camera][0], poses_dict[anchor_camera][1])
        C_scale = camera_center(poses_dict[scale_camera][0], poses_dict[scale_camera][1])
        d0 = float(np.linalg.norm(C_scale - C_anchor))
        if d0 < 1e-6:
            d0 = 1.0
    elif initial_baseline is not None:
        d0 = float(initial_baseline)
    else:
        d0 = 1.0

    use_scale_gauge = (
        scale_camera is not None
        and scale_camera in cam_slot
        and config.scale_constraint_weight > 0
    )

    def unpack(params):
        rvecs = np.empty((n_obs, 3))
        ts = np.empty((n_obs, 3))
        cams = params[:n_cam_params].reshape(-1, 6) if n_cam_params > 0 else np.zeros((0, 6))
        for c in cam_ids:
            m = obs_cam == c
            if not np.any(m):
                continue
            if c in fixed_cams:
                rvecs[m], ts[m] = fixed_rvec[c], fixed_t[c]
            else:
                slot = cam_slot[c]
                rvecs[m], ts[m] = cams[slot, :3], cams[slot, 3:]
        pts = params[n_cam_params:].reshape(-1, 3)[obs_pt]
        return rvecs, ts, pts

    def resid(params):
        rvecs, ts, pts = unpack(params)
        Xc = _rotate(rvecs, pts) + ts
        # Reprojection residuals
        z = np.maximum(Xc[:, 2], 1e-12)
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        res_reproj = np.concatenate([u - obs_xy[:, 0], v - obs_xy[:, 1]])

        if use_scale_gauge:
            cams = params[:n_cam_params].reshape(-1, 6)
            slot = cam_slot[scale_camera]
            R_scale = rodrigues_to_R(cams[slot, :3])
            t_scale = cams[slot, 3:]
            C_scale = -R_scale.T @ t_scale

            R_anchor = rodrigues_to_R(fixed_rvec[anchor_camera])
            t_anchor = fixed_t[anchor_camera]
            C_anchor = -R_anchor.T @ t_anchor

            current_baseline = float(np.linalg.norm(C_scale - C_anchor))
            res_scale = np.array([config.scale_constraint_weight * (current_baseline - d0)])
            return np.concatenate([res_reproj, res_scale])

        return res_reproj

    # Jacobian sparsity structure
    n_residuals = 2 * n_obs + (1 if use_scale_gauge else 0)
    sparsity = lil_matrix((n_residuals, n_cam_params + 3 * n_pts), dtype=int)
    rows = np.arange(n_obs)

    for c in free_cams:
        m = np.where(obs_cam == c)[0]
        slot = cam_slot[c]
        for k in range(6):
            sparsity[m, 6 * slot + k] = 1
            sparsity[n_obs + m, 6 * slot + k] = 1

    for k in range(3):
        sparsity[rows, n_cam_params + 3 * obs_pt + k] = 1
        sparsity[n_obs + rows, n_cam_params + 3 * obs_pt + k] = 1

    if use_scale_gauge:
        scale_slot = cam_slot[scale_camera]
        scale_row = 2 * n_obs
        # Scale depends on rotation and translation of scale camera
        for k in range(6):
            sparsity[scale_row, 6 * scale_slot + k] = 1

    r0 = resid(x0)
    rms_before = float(np.sqrt(np.mean(r0[:n_obs] ** 2 + r0[n_obs : 2 * n_obs] ** 2)))

    sol = least_squares(
        resid,
        x0,
        jac_sparsity=sparsity,
        method="trf",
        loss="soft_l1",
        f_scale=config.ba_robust_scale_px,
        x_scale="jac",
        ftol=1e-6,
        verbose=verbose,
    )

    r1 = sol.fun
    rms_after = float(np.sqrt(np.mean(r1[:n_obs] ** 2 + r1[n_obs : 2 * n_obs] ** 2)))

    new_poses: dict[int, tuple[np.ndarray, np.ndarray]] = dict(poses_dict)
    if n_cam_params > 0:
        cams = sol.x[:n_cam_params].reshape(-1, 6)
        for c in free_cams:
            slot = cam_slot[c]
            R_new = rodrigues_to_R(cams[slot, :3])
            t_new = cams[slot, 3:]
            new_poses[c] = (R_new, t_new)

    new_points = sol.x[n_cam_params:].reshape(-1, 3)
    return new_poses, new_points, rms_before, rms_after
