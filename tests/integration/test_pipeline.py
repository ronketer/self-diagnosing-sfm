import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.config import SfMConfig
from src.dataset import load_dataset
from src.evaluation import evaluate
from src.features import build_match_graph, detect_features
from src.sfm import run_sfm


def all_valid_rotations(recon) -> bool:
    """Verify that all estimated camera rotations are valid SO(3) matrices."""
    for img_id, (R, _) in recon.poses.items():
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4):
            return False
        if not np.isclose(np.linalg.det(R), 1.0, atol=1e-4):
            return False
    return True


def all_landmarks_have_observations(recon, tm) -> bool:
    """Verify that every reconstructed 3D point maps back to a valid feature track with >= 2 views."""
    pt_ids_in_tracks = set()
    for tr in tm.tracks:
        if tr.point3d_id is not None:
            pt_ids_in_tracks.add(tr.point3d_id)
            if len(tr.obs) < 2:
                return False
    return len(pt_ids_in_tracks) == len(recon.points3d)


def test_temple_sparse_ring_reconstruction():
    root = Path(__file__).parent.parent.parent
    data_dir = root / "data" / "templeSparseRing"
    if not data_dir.exists():
        print(f"Skipping integration test: {data_dir} does not exist.")
        return

    images, gts, K = load_dataset(data_dir)
    keypoints, descriptors = [], []
    for im in images:
        kp, desc = detect_features(im.gray)
        keypoints.append(kp)
        descriptors.append(desc)

    cache_dir = root / "output" / "templeSparseRing" / "cache"
    cache = cache_dir / "matches_templeSparseRing_all.npz"
    graph = build_match_graph(keypoints, descriptors, cache_path=cache)

    config = SfMConfig()
    recon, tm, stats = run_sfm(
        keypoints,
        graph,
        K,
        config=config,
        use_ba=True,
        use_geometric_verification=True,
        log=lambda s: None,
    )

    # Invariants checks
    assert stats["n_registered"] >= 4, f"Registered only {stats['n_registered']} cameras"
    assert stats["n_points"] >= 100, f"Reconstructed only {stats['n_points']} landmarks"
    assert stats["rms_final"] is not None and stats["rms_final"] < 3.0, (
        f"High reprojection RMS: {stats['rms_final']}"
    )
    assert all_valid_rotations(recon), "Invalid rotation matrix detected"
    assert all_landmarks_have_observations(recon, tm), "Orphan landmarks detected without tracks"

    out_dir = root / "output" / "test_integration_templeSparseRing"
    out_dir.mkdir(parents=True, exist_ok=True)
    if gts is not None:
        metrics = evaluate(recon, gts, stats, out_dir)
        assert metrics["rotation_error_deg"]["mean"] < 5.0, "High mean rotation error"
        assert metrics["camera_center_error"]["mean_pct_of_diameter"] < 1.0, "High center error"

    assert (out_dir / "diagnostics.json").exists()
    assert (out_dir / "reconstruction.json").exists()


if __name__ == "__main__":
    test_temple_sparse_ring_reconstruction()
    print("test_pipeline integration: OK")
