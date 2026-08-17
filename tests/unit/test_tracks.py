"""Unit tests for conflict-aware track merging and track invariants."""

import numpy as np
from src.tracks import ConflictAwareUnionFind, TrackManager


def test_conflict_aware_union_find():
    uf = ConflictAwareUnionFind()

    # Image 0 kp 10 <-> Image 1 kp 20
    assert uf.union((0, 10), (1, 20)) is True

    # Image 1 kp 20 <-> Image 0 kp 40  (Conflict! Both have an observation in image 0)
    assert uf.union((1, 20), (0, 40)) is False

    # Image 1 kp 20 <-> Image 2 kp 30  (Valid extension to image 2)
    assert uf.union((1, 20), (2, 30)) is True

    # Check root component of (0, 10)
    root = uf.find((0, 10))
    assert uf.images_per_root[root] == {0, 1, 2}
    assert uf.find((1, 20)) == root
    assert uf.find((2, 30)) == root
    assert uf.find((0, 40)) != root


def test_track_manager_invariant():
    # Build a match graph with a conflicting edge
    # (0, 10) matches (1, 20)
    # (1, 20) matches (0, 40) -- conflicting!
    # (1, 20) matches (2, 30)
    match_graph = {
        (0, 1): np.array([[10, 20], [40, 20]]),
        (1, 2): np.array([[20, 30]]),
    }

    tm = TrackManager(match_graph)
    assert tm.n_rejected_edges == 1
    assert len(tm.tracks) == 1

    track = tm.tracks[0]
    # Track must have at most one observation per image
    assert len(track.obs) == 3
    assert track.obs[0] == 10
    assert track.obs[1] == 20
    assert track.obs[2] == 30

    # Invariant holds across all tracks
    for tr in tm.tracks:
        seen_images = set()
        for img in tr.obs:
            assert img not in seen_images
            seen_images.add(img)


if __name__ == "__main__":
    test_conflict_aware_union_find()
    test_track_manager_invariant()
    print("test_tracks: OK")
