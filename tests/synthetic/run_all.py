"""Convenience runner for the complete synthetic/unit SfM regression suite.

Run from the repository root with:

    uv run python -m tests.synthetic.run_all

This file is intentionally named ``run_all.py`` rather than ``test_*.py``
so pytest-style discovery does not execute every test twice.
"""

from tests.synthetic.test_degeneracies import (
    test_homography_vs_essential_evidence,
    test_planar_scene_degeneracy,
    test_pnp_quality_gates,
    test_pnp_with_gross_outliers,
    test_points_behind_camera,
    test_pure_rotation_and_zero_baseline,
    test_shared_intrinsics_and_distortion_validation,
)
from tests.synthetic.test_disconnected_graph import (
    test_disconnected_view_graph,
)
from tests.synthetic.test_synthetic_sfm import (
    test_bundle_adjustment,
    test_bundle_adjustment_gauge_invariants,
    test_essential_and_pose,
    test_pnp_pose_recovery,
    test_rodrigues_roundtrip,
    test_triangulation,
)
from tests.unit.test_essential import (
    test_deterministic_essential_ransac,
    test_essential_matrix_properties,
    test_hartley_normalize,
    test_pair_geometry_reuse_in_initialization,
)
from tests.unit.test_initialization import (
    test_initial_pair_expansion_support,
    test_initial_pair_prefers_expandable_pair_over_isolated_high_parallax_pair,
    test_initial_pair_without_tracks_falls_back_to_geometry,
)
from tests.unit.test_projection import (
    test_camera_center,
    test_point_depths,
    test_to_normalized_and_reproject,
    test_triangulation_angles,
)
from tests.unit.test_tracks import (
    test_confidence_ordered_track_merging,
    test_conflict_aware_union_find,
    test_track_manager_invariant,
)
from tests.unit.test_triangulation import (
    test_multiview_triangulation_best_parallax_pair,
    test_triangulate_track_validation,
    test_two_view_and_multi_view_dlt,
)


def run_core_synthetic_tests() -> None:
    """Core geometry, PnP and BA tests."""

    test_rodrigues_roundtrip()
    print("rodrigues roundtrip: OK")

    test_triangulation()
    print("triangulation: OK")

    test_essential_and_pose()
    print("essential + pose recovery: OK")

    test_pnp_pose_recovery()
    print("PnP pose recovery: OK")

    test_bundle_adjustment()
    test_bundle_adjustment_gauge_invariants()
    print("bundle adjustment + gauge: OK")


def run_unit_tests() -> None:
    """Low-level mathematical and state-management tests."""

    # Projection.
    test_to_normalized_and_reproject()
    test_camera_center()
    test_point_depths()
    test_triangulation_angles()

    # Essential geometry.
    test_hartley_normalize()
    test_essential_matrix_properties()
    test_deterministic_essential_ransac()
    test_pair_geometry_reuse_in_initialization()

    # Initialization policy.
    test_initial_pair_expansion_support()
    test_initial_pair_prefers_expandable_pair_over_isolated_high_parallax_pair()
    test_initial_pair_without_tracks_falls_back_to_geometry()

    # Triangulation.
    test_two_view_and_multi_view_dlt()
    test_triangulate_track_validation()
    test_multiview_triangulation_best_parallax_pair()

    # Tracks.
    test_conflict_aware_union_find()
    test_track_manager_invariant()
    test_confidence_ordered_track_merging()

    print("unit tests: OK")


def run_degeneracy_and_graph_tests() -> None:
    """Controlled failure, degeneracy, and graph-behavior tests."""

    test_pure_rotation_and_zero_baseline()
    test_homography_vs_essential_evidence()
    test_planar_scene_degeneracy()
    test_points_behind_camera()

    test_pnp_with_gross_outliers()
    test_pnp_quality_gates()

    test_shared_intrinsics_and_distortion_validation()

    test_disconnected_view_graph()

    print("degeneracy & graph tests: OK")


def main() -> None:
    run_core_synthetic_tests()
    run_unit_tests()
    run_degeneracy_and_graph_tests()

    print(
        "all synthetic and unit tests passed"
    )


if __name__ == "__main__":
    main()
