"""Unit tests for canonical, conflict-aware, confidence-ordered feature tracks."""

import numpy as np

from src.models import Observation, Track
from src.tracks import ConflictAwareUnionFind, TrackManager


def test_conflict_aware_union_find():
    uf = ConflictAwareUnionFind()

    assert uf.union((0, 10), (1, 20)) is True
    assert uf.union((1, 20), (0, 40)) is False
    assert uf.union((1, 20), (2, 30)) is True

    root = uf.find((0, 10))
    assert uf.images_per_root[root] == {0, 1, 2}
    assert uf.find((1, 20)) == root
    assert uf.find((2, 30)) == root
    assert uf.find((0, 40)) != root


def test_track_manager_uses_canonical_track_model():
    manager = TrackManager(
        {(0, 1): np.array([[10, 20]])}
    )

    assert len(manager.tracks) == 1
    track = manager.tracks[0]

    assert isinstance(track, Track)
    assert isinstance(track.observations[0], Observation)
    assert isinstance(track.observations[1], Observation)


def test_track_manager_invariant():
    match_graph = {
        (0, 1): np.array(
            [
                [10, 20],
                [40, 20],
            ]
        ),
        (1, 2): np.array([[20, 30]]),
    }

    manager = TrackManager(match_graph)

    assert manager.n_rejected_edges == 1
    assert len(manager.tracks) == 1

    track = manager.tracks[0]

    assert track.image_ids == {0, 1, 2}
    assert len(track.observations) == 3
    assert track.keypoint_id(0) == 10
    assert track.keypoint_id(1) == 20
    assert track.keypoint_id(2) == 30

    for image_id, observation in track.observations.items():
        assert observation.image_id == image_id


def test_confidence_ordered_track_merging():
    match_graph = {
        (0, 1): np.array(
            [
                [40, 20],  # weak conflicting edge is intentionally first
                [10, 20],  # strong correct edge
            ]
        ),
        (1, 2): np.array([[20, 30]]),
    }

    edge_scores = {
        (0, 1): np.array([0.10, 0.95]),
        (1, 2): np.array([0.80]),
    }

    manager = TrackManager(
        match_graph,
        edge_confidences=edge_scores,
    )

    assert manager.n_rejected_edges == 1
    assert len(manager.tracks) == 1

    track = manager.tracks[0]
    assert track.keypoint_id(0) == 10
    assert track.keypoint_id(1) == 20
    assert track.keypoint_id(2) == 30


def test_confidence_count_mismatch_is_rejected():
    try:
        TrackManager(
            {
                (0, 1): np.array(
                    [
                        [10, 20],
                        [11, 21],
                    ]
                )
            },
            edge_confidences={
                (0, 1): np.array([0.9])
            },
        )
    except ValueError as exc:
        assert "Confidence count mismatch" in str(exc)
    else:
        raise AssertionError(
            "mismatched edge-confidence length should fail"
        )


def test_correspondences_2d3d_use_typed_observation():
    manager = TrackManager(
        {(0, 1): np.array([[1, 2]])}
    )
    manager.tracks[0].point3d_id = 0

    keypoints = [
        np.array(
            [
                [10.0, 20.0],
                [30.0, 40.0],
            ]
        ),
        np.array(
            [
                [50.0, 60.0],
                [70.0, 80.0],
                [90.0, 100.0],
            ]
        ),
    ]

    points3d = np.array([[1.0, 2.0, 3.0]])

    pts2d, pts3d, track_ids = manager.correspondences_2d3d(
        1,
        keypoints,
        points3d,
    )

    np.testing.assert_allclose(
        pts2d,
        [[90.0, 100.0]],
    )
    np.testing.assert_allclose(
        pts3d,
        [[1.0, 2.0, 3.0]],
    )
    np.testing.assert_array_equal(
        track_ids,
        [0],
    )


if __name__ == "__main__":
    test_conflict_aware_union_find()
    test_track_manager_uses_canonical_track_model()
    test_track_manager_invariant()
    test_confidence_ordered_track_merging()
    test_confidence_count_mismatch_is_rejected()
    test_correspondences_2d3d_use_typed_observation()
    print("test_tracks.py: ALL PASSED")
