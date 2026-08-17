"""Synthetic test for disconnected view-graph handling and NO_2D3D_BRIDGE termination."""

import numpy as np
from src.bundle_adjustment import rodrigues_to_R
from src.config import SfMConfig
from src.geometry.projection import reproject
from src.sfm import run_sfm
from src.validation.diagnostics import (
    CameraRegistrationStatus,
    ReconstructionStatus,
    TerminationReason,
)


def test_disconnected_view_graph():
    """Two disjoint image clusters (0,1,2) and (3,4) with no shared correspondences.

    Expectation:
    - Images {0, 1, 2} in Component A are registered.
    - Images {3, 4} in Component B remain unregistered and blocked with 0 correspondences.
    - Reconstruction status is PARTIAL.
    - Termination reason is NO_2D3D_BRIDGE.
    - View graph diagnostics report 2 components.
    """
    rng = np.random.default_rng(999)
    K = np.array([[1000.0, 0, 500], [0, 1000.0, 500], [0, 0, 1]], dtype=float)

    # Cluster A: 3D points A (Z in [4, 7])
    X_A = rng.uniform([-1, -1, 4], [1, 1, 7], size=(60, 3))
    # Cluster B: 3D points B (Z in [10, 14], shifted)
    X_B = rng.uniform([10, 10, 10], [12, 12, 14], size=(60, 3))

    poses_A = {
        0: (np.eye(3), np.zeros(3)),
        1: (rodrigues_to_R(np.array([0.05, 0.0, 0.0])), np.array([0.5, 0.0, 0.0])),
        2: (rodrigues_to_R(np.array([-0.05, 0.0, 0.0])), np.array([-0.5, 0.0, 0.0])),
    }

    poses_B = {
        3: (np.eye(3), np.array([10.0, 0.0, 0.0])),
        4: (rodrigues_to_R(np.array([0.02, 0.0, 0.0])), np.array([10.5, 0.0, 0.0])),
    }

    # Generate 2D observations
    keypoints = []
    # Images 0, 1, 2 observe X_A
    for i in range(3):
        R, t = poses_A[i]
        keypoints.append(reproject(K, R, t, X_A))

    # Images 3, 4 observe X_B
    for i in (3, 4):
        R, t = poses_B[i]
        keypoints.append(reproject(K, R, t, X_B))

    # Construct match graph: ONLY intra-cluster matches, 0 cross-cluster matches
    match_graph = {}
    # Cluster A pairwise matches
    for i in range(3):
        for j in range(i + 1, 3):
            m = np.column_stack([np.arange(len(X_A)), np.arange(len(X_A))])
            match_graph[(i, j)] = m

    # Cluster B pairwise matches
    match_graph[(3, 4)] = np.column_stack([np.arange(len(X_B)), np.arange(len(X_B))])

    # Run SfM
    cfg = SfMConfig(
        min_verified_matches=20,
        min_init_inliers=30,
        min_pnp_correspondences=15,
        min_pnp_inliers=10,
    )
    recon, tm, stats = run_sfm(
        keypoints,
        match_graph,
        K,
        config=cfg,
        use_ba=True,
        log=lambda s: None,  # Suppress stdout during test
    )

    diag = stats["diagnostics"]

    # Assertions
    assert diag.status == ReconstructionStatus.PARTIAL
    assert diag.termination_reason == TerminationReason.NO_2D3D_BRIDGE

    # Check registered cameras
    assert set(recon.cameras) == {0, 1, 2}
    assert stats["n_registered"] == 3
    assert stats["n_images"] == 5

    # Check view graph components
    vg = diag.view_graph
    assert vg["num_components"] == 2
    assert sorted(vg["active_component"]) == [0, 1, 2]
    assert [3, 4] in vg["other_components"] or [4, 3] in vg["other_components"]

    # Check per-camera status
    for i in (0, 1, 2):
        assert diag.camera_status[str(i)]["status"] == CameraRegistrationStatus.REGISTERED.value
    for j in (3, 4):
        cam_info = diag.camera_status[str(j)]
        assert cam_info["status"] == CameraRegistrationStatus.BLOCKED_NO_2D3D_CORRESPONDENCES.value
        assert cam_info["last_2d3d_correspondences"] == 0


if __name__ == "__main__":
    test_disconnected_view_graph()
    print("test_disconnected_graph.py: ALL PASSED")
