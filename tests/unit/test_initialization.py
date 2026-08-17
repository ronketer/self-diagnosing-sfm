"""Unit tests for incremental-SfM initialization-pair selection."""

from src.initialization import (
    InitialPairEvaluation,
    choose_best_initial_pair_evaluation,
    initial_pair_expansion_support,
)
from src.models import Observation, Track


def _make_three_view_track(
    track_id: int,
    image_a: int,
    image_b: int,
    image_c: int,
) -> Track:
    """Create a simple three-view feature track."""

    return Track(
        id=track_id,
        observations={
            image_a: Observation(
                image_id=image_a,
                keypoint_id=track_id * 10 + 0,
            ),
            image_b: Observation(
                image_id=image_b,
                keypoint_id=track_id * 10 + 1,
            ),
            image_c: Observation(
                image_id=image_c,
                keypoint_id=track_id * 10 + 2,
            ),
        },
    )


def _make_two_view_track(
    track_id: int,
    image_a: int,
    image_b: int,
) -> Track:
    """Create a track with no third-camera expansion support."""

    return Track(
        id=track_id,
        observations={
            image_a: Observation(
                image_id=image_a,
                keypoint_id=track_id * 10 + 0,
            ),
            image_b: Observation(
                image_id=image_b,
                keypoint_id=track_id * 10 + 1,
            ),
        },
    )


def test_initial_pair_expansion_support():
    """Tracks visible in A+B+C should create potential PnP support for C."""

    tracks = [
        _make_three_view_track(
            track_id=i,
            image_a=0,
            image_b=1,
            image_c=2,
        )
        for i in range(20)
    ]

    (
        expandable_count,
        best_next_image,
        best_next_correspondences,
    ) = initial_pair_expansion_support(
        image_a=0,
        image_b=1,
        tracks=tracks,
        min_correspondences=15,
    )

    assert expandable_count == 1
    assert best_next_image == 2
    assert best_next_correspondences == 20


def test_initial_pair_prefers_expandable_pair_over_isolated_high_parallax_pair():
    """Incremental SfM must prefer an expandable valid pair.

    Pair (0, 1):
        lower parallax = 15 degrees
        20 tracks extend into image 2

    Pair (3, 4):
        higher parallax = 30 degrees
        isolated two-view component

    The old "maximum parallax wins" rule would select (3, 4).

    The incremental-SfM rule must select (0, 1).
    """

    expandable_tracks = [
        _make_three_view_track(
            track_id=i,
            image_a=0,
            image_b=1,
            image_c=2,
        )
        for i in range(20)
    ]

    isolated_tracks = [
        _make_two_view_track(
            track_id=100 + i,
            image_a=3,
            image_b=4,
        )
        for i in range(40)
    ]

    tracks = (
        expandable_tracks
        + isolated_tracks
    )

    expandable_pair = InitialPairEvaluation(
        image_a=0,
        image_b=1,
        n_matches=200,
        n_inliers=150,
        inlier_ratio=0.75,
        cheirality_ratio=1.0,
        median_parallax_deg=15.0,
        median_reprojection_px=0.5,
        accepted=True,
        score=15.0,
    )

    isolated_high_parallax_pair = InitialPairEvaluation(
        image_a=3,
        image_b=4,
        n_matches=300,
        n_inliers=250,
        inlier_ratio=0.83,
        cheirality_ratio=1.0,
        median_parallax_deg=30.0,
        median_reprojection_px=0.5,
        accepted=True,
        score=30.0,
    )

    selected = choose_best_initial_pair_evaluation(
        [
            isolated_high_parallax_pair,
            expandable_pair,
        ],
        tracks=tracks,
        min_pnp_correspondences=15,
    )

    assert (
        selected.image_a,
        selected.image_b,
    ) == (
        0,
        1,
    )

    assert selected.expandable_camera_count == 1
    assert selected.best_next_image == 2
    assert selected.best_next_correspondences == 20

    assert (
        isolated_high_parallax_pair.expandable_camera_count
        == 0
    )

    assert (
        isolated_high_parallax_pair.best_next_image
        is None
    )


def test_initial_pair_without_tracks_falls_back_to_geometry():
    """Without track information the selection remains geometry-based."""

    lower_parallax_pair = InitialPairEvaluation(
        image_a=0,
        image_b=1,
        n_matches=200,
        n_inliers=150,
        inlier_ratio=0.75,
        cheirality_ratio=1.0,
        median_parallax_deg=15.0,
        median_reprojection_px=0.5,
        accepted=True,
        score=15.0,
    )

    higher_parallax_pair = InitialPairEvaluation(
        image_a=3,
        image_b=4,
        n_matches=300,
        n_inliers=250,
        inlier_ratio=0.83,
        cheirality_ratio=1.0,
        median_parallax_deg=30.0,
        median_reprojection_px=0.5,
        accepted=True,
        score=30.0,
    )

    selected = choose_best_initial_pair_evaluation(
        [
            lower_parallax_pair,
            higher_parallax_pair,
        ],
        tracks=None,
        min_pnp_correspondences=15,
    )

    assert (
        selected.image_a,
        selected.image_b,
    ) == (
        3,
        4,
    )


if __name__ == "__main__":
    test_initial_pair_expansion_support()
    test_initial_pair_prefers_expandable_pair_over_isolated_high_parallax_pair()
    test_initial_pair_without_tracks_falls_back_to_geometry()

    print(
        "test_initialization.py: ALL PASSED"
    )