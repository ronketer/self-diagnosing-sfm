"""Synthetic sanity checks for core geometry, PnP, and bundle adjustment."""

import numpy as np

from src.bundle_adjustment import (
    R_to_rodrigues,
    rodrigues_to_R,
    run_bundle_adjustment,
)
from src.config import SfMConfig
from src.geometry.essential import (
    choose_pose,
    decompose_essential,
    ransac_essential,
)
from src.geometry.projection import reproject
from src.pnp import estimate_camera_pose, ransac_pnp
from src.triangulation import triangulate_two_view

rng = np.random.default_rng(42)


def random_rotation(scale=0.3):
    return rodrigues_to_R(rng.normal(scale=scale, size=3))


def make_scene(n_pts=200):
    K = np.array([[1500.0, 0, 320], [0, 1500.0, 240], [0, 0, 1]])
    X = rng.uniform([-1, -1, 4], [1, 1, 8], size=(n_pts, 3))
    R1, t1 = np.eye(3), np.zeros(3)
    R2 = random_rotation(0.1)
    t2 = np.array([1.0, 0.1, 0.05])
    return K, X, (R1, t1), (R2, t2)


def test_rodrigues_roundtrip():
    for _ in range(100):
        R = random_rotation(scale=1.5)
        assert np.allclose(rodrigues_to_R(R_to_rodrigues(R)), R, atol=1e-9)


def test_essential_and_pose():
    K, X, (R1, t1), (R2, t2) = make_scene()
    x1 = reproject(K, R1, t1, X)
    x2 = reproject(K, R2, t2, X)
    Ki = np.linalg.inv(K)
    x1n = (np.hstack([x1, np.ones((len(x1), 1))]) @ Ki.T)[:, :2]
    x2n = (np.hstack([x2, np.ones((len(x2), 1))]) @ Ki.T)[:, :2]

    E, inl = ransac_essential(x1n, x2n)
    assert inl.sum() > 0.95 * len(X), f"only {inl.sum()} inliers"

    R, t, mask = choose_pose(decompose_essential(E), x1n, x2n)
    assert np.allclose(R, R2, atol=1e-4), "rotation mismatch"
    t_gt = t2 / np.linalg.norm(t2)
    assert np.allclose(np.abs(t @ t_gt), 1.0, atol=1e-4), "translation dir mismatch"


def test_triangulation():
    K, X, (R1, t1), (R2, t2) = make_scene()
    x1n = reproject(np.eye(3), R1, t1, X)
    x2n = reproject(np.eye(3), R2, t2, X)
    P1 = np.hstack([R1, t1.reshape(3, 1)])
    P2 = np.hstack([R2, t2.reshape(3, 1)])
    Xt = triangulate_two_view(P1, P2, x1n, x2n)
    assert np.allclose(Xt, X, atol=1e-8), f"max err {np.abs(Xt - X).max()}"


def test_opencv_and_scratch_pnp():
    K, X, _, (R2, t2) = make_scene()
    x2 = reproject(K, R2, t2, X)

    # Scratch reference PnP
    res_scratch = ransac_pnp(X, x2, K)
    assert res_scratch is not None
    R_sc, t_sc, mask_sc = res_scratch
    assert np.allclose(R_sc, R2, atol=1e-5), "Scratch PnP rotation mismatch"
    assert np.allclose(t_sc, t2, atol=1e-5), "Scratch PnP translation mismatch"

    # OpenCV production PnP
    res_cv = estimate_camera_pose(X, x2, K, SfMConfig())
    assert res_cv.accepted
    assert res_cv.pose is not None
    assert np.allclose(res_cv.pose.R, R2, atol=1e-5), "OpenCV PnP rotation mismatch"
    assert np.allclose(res_cv.pose.t, t2, atol=1e-5), "OpenCV PnP translation mismatch"
    assert res_cv.reprojection_rms_px is not None and res_cv.reprojection_rms_px < 1e-3


def test_bundle_adjustment():
    K, X, (R1, t1), (R2, t2) = make_scene(n_pts=100)
    R3 = rodrigues_to_R(np.array([0.05, -0.2, 0.02]))
    t3 = np.array([-0.8, 0.05, 0.1])
    cams = {0: (R1, t1), 1: (R2, t2), 2: (R3, t3)}
    obs = []
    for c, (R, t) in cams.items():
        x = reproject(K, R, t, X)
        for j in range(len(X)):
            obs.append((c, j, x[j]))

    # Perturb poses and landmarks
    poses_pert = {0: (R1, t1)}
    for c in (1, 2):
        R, t = cams[c]
        Rp = rodrigues_to_R(R_to_rodrigues(R) + rng.normal(scale=0.01, size=3))
        poses_pert[c] = (Rp, t + rng.normal(scale=0.02, size=3))
    X_pert = X + rng.normal(scale=0.02, size=X.shape)

    poses_opt, X_opt, rms_before, rms_after = run_bundle_adjustment(
        K, poses_pert, X_pert, obs, anchor_camera=0, scale_camera=1
    )
    assert rms_before > 1.0, f"perturbation too small ({rms_before})"
    assert rms_after < 1e-4, f"BA did not converge (rms {rms_after})"


if __name__ == "__main__":
    test_rodrigues_roundtrip()
    test_triangulation()
    test_essential_and_pose()
    test_opencv_and_scratch_pnp()
    test_bundle_adjustment()
    print("test_synthetic_sfm: OK")
