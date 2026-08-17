"""Synthetic sanity checks and comprehensive test suite for 3D reconstruction."""

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
from tests.synthetic.test_degeneracies import (
    test_planar_scene_degeneracy,
    test_pnp_with_gross_outliers,
    test_points_behind_camera,
    test_pure_rotation_and_zero_baseline,
)
from tests.unit.test_essential import (
    test_essential_matrix_properties,
    test_hartley_normalize,
)
from tests.unit.test_projection import (
    test_camera_center,
    test_point_depths,
    test_to_normalized_and_reproject,
    test_triangulation_angles,
)
from tests.unit.test_tracks import (
    test_conflict_aware_union_find,
    test_track_manager_invariant,
)
from tests.unit.test_triangulation import (
    test_triangulate_track_validation,
    test_two_view_and_multi_view_dlt,
)

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
    print("rodrigues roundtrip: OK")


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
    # relative pose: R_rel = R2 (since cam1 = identity), t up to scale
    assert np.allclose(R, R2, atol=1e-4), "rotation mismatch"
    t_gt = t2 / np.linalg.norm(t2)
    assert np.allclose(np.abs(t @ t_gt), 1.0, atol=1e-4), "translation dir mismatch"
    print("essential + pose recovery: OK")


def test_triangulation():
    K, X, (R1, t1), (R2, t2) = make_scene()
    x1n = reproject(np.eye(3), R1, t1, X)
    x2n = reproject(np.eye(3), R2, t2, X)
    P1 = np.hstack([R1, t1.reshape(3, 1)])
    P2 = np.hstack([R2, t2.reshape(3, 1)])
    Xt = triangulate_two_view(P1, P2, x1n, x2n)
    assert np.allclose(Xt, X, atol=1e-8), f"max err {np.abs(Xt - X).max()}"
    print("triangulation: OK")


def test_pnp():
    K, X, _, (R2, t2) = make_scene()
    x2 = reproject(K, R2, t2, X)

    # Scratch reference PnP
    res_sc = ransac_pnp(X, x2, K)
    assert res_sc is not None
    R_sc, t_sc, mask_sc = res_sc
    assert np.allclose(R_sc, R2, atol=1e-5), "Scratch PnP rotation mismatch"
    assert np.allclose(t_sc, t2, atol=1e-5), f"Scratch PnP translation mismatch: {t_sc} vs {t2}"

    # OpenCV production PnP
    res_cv = estimate_camera_pose(X, x2, K, SfMConfig())
    assert res_cv.accepted, f"OpenCV PnP rejected: {res_cv.reason}"
    assert res_cv.pose is not None
    assert np.allclose(res_cv.pose.R, R2, atol=1e-5), "OpenCV PnP rotation mismatch"
    assert np.allclose(res_cv.pose.t, t2, atol=1e-5), f"OpenCV PnP translation mismatch: {res_cv.pose.t} vs {t2}"
    print("PnP (OpenCV + scratch): OK")


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

    # perturb
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
    print(f"bundle adjustment: OK (rms {rms_before:.2f} -> {rms_after:.2e} px)")


def run_unit_tests():
    test_to_normalized_and_reproject()
    test_camera_center()
    test_point_depths()
    test_triangulation_angles()
    test_hartley_normalize()
    test_essential_matrix_properties()
    test_two_view_and_multi_view_dlt()
    test_triangulate_track_validation()
    test_conflict_aware_union_find()
    test_track_manager_invariant()
    print("unit tests: OK")


def run_degeneracy_tests():
    test_pure_rotation_and_zero_baseline()
    test_planar_scene_degeneracy()
    test_points_behind_camera()
    test_pnp_with_gross_outliers()
    print("degeneracy tests: OK")


if __name__ == "__main__":
    test_rodrigues_roundtrip()
    test_triangulation()
    test_essential_and_pose()
    test_pnp()
    test_bundle_adjustment()
    run_unit_tests()
    run_degeneracy_tests()
    print("all synthetic and unit tests passed")
