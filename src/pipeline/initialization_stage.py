"""Pipeline-level two-view initialization and creation of the first map."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..config import SfMConfig
from ..initialization import (
    InitialPairEvaluation,
    choose_best_initial_pair_evaluation,
    evaluate_pair,
    evaluate_pair_geometry,
    select_init_pair,
)
from ..models import (
    CameraIntrinsics,
    CameraPose,
    PairGeometry,
    Reconstruction,
    ReconstructionState,
)
from ..tracks import TrackManager
from ..validation.diagnostics import (
    ReconstructionDiagnostics,
    ReconstructionStatus,
    TerminationReason,
)
from .diagnostics import build_failure_stats, compute_view_graph_components
from .reconstruction_ops import triangulate_new_landmarks


@dataclass
class InitializationResult:
    """Result of the two-view initialization pipeline stage."""

    state: ReconstructionState | None = None
    image_0: int | None = None
    image_1: int | None = None
    failure_stats: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        return self.state is not None


def initialize_reconstruction(
    *,
    intrinsics: CameraIntrinsics,
    keypoints: list[np.ndarray],
    match_graph: dict[tuple[int, int], np.ndarray],
    pair_geometries: dict[tuple[int, int], PairGeometry],
    verified_match_graph: dict[tuple[int, int], np.ndarray],
    track_manager: TrackManager,
    diagnostics: ReconstructionDiagnostics,
    config: SfMConfig,
    raw_pairs: int,
    raw_matches: int,
    use_ba: bool = True,
    use_geometric_verification: bool = True,
    log: Callable[[str], None] = print,
) -> InitializationResult:
    """Select initial pair, establish gauge, and triangulate initial landmarks."""
    K_mat = intrinsics.K
    reconstruction = Reconstruction(intrinsics=intrinsics)

    if use_geometric_verification and not pair_geometries:
        diagnostics.status = ReconstructionStatus.FAILED
        diagnostics.termination_reason = (
            TerminationReason.INSUFFICIENT_VERIFIED_GEOMETRY
        )
        diagnostics.view_graph = compute_view_graph_components(
            len(keypoints),
            [],
            set(),
        )

        stats = build_failure_stats(
            diagnostics=diagnostics,
            n_images=len(keypoints),
            raw_pairs=raw_pairs,
            raw_matches=raw_matches,
            verified_pairs=0,
            verified_matches=0,
            use_ba=use_ba,
            use_geometric_verification=use_geometric_verification,
        )
        return InitializationResult(failure_stats=stats)

    init_evaluations: list[InitialPairEvaluation] = []

    try:
        if use_geometric_verification:
            best_init, init_evaluations = select_init_pair(
                pair_geometries,
                keypoints,
                K_mat,
                config,
                tracks=track_manager.tracks,
            )
        else:
            init_evaluations = [
                evaluate_pair(
                    image_a,
                    image_b,
                    matches,
                    keypoints,
                    K_mat,
                    config,
                )
                for (image_a, image_b), matches in match_graph.items()
            ]

            best_init = choose_best_initial_pair_evaluation(
                init_evaluations,
                tracks=track_manager.tracks,
                min_pnp_correspondences=config.min_pnp_correspondences,
            )

    except RuntimeError:
        if use_geometric_verification and pair_geometries:
            init_evaluations = [
                evaluate_pair_geometry(
                    geometry,
                    keypoints,
                    K_mat,
                    config,
                )
                for geometry in pair_geometries.values()
            ]

        for evaluation in init_evaluations:
            diagnostics.record_pair_evaluation(evaluation.to_dict())

        diagnostics.status = ReconstructionStatus.FAILED
        diagnostics.termination_reason = (
            TerminationReason.INITIALIZATION_FAILED
        )
        diagnostics.view_graph = compute_view_graph_components(
            len(keypoints),
            list(verified_match_graph.keys()),
            set(),
        )

        stats = build_failure_stats(
            diagnostics=diagnostics,
            n_images=len(keypoints),
            raw_pairs=raw_pairs,
            raw_matches=raw_matches,
            verified_pairs=len(verified_match_graph),
            verified_matches=sum(
                len(matches) for matches in verified_match_graph.values()
            ),
            use_ba=use_ba,
            use_geometric_verification=use_geometric_verification,
        )
        return InitializationResult(failure_stats=stats)

    image_0 = best_init.image_a
    image_1 = best_init.image_b

    if best_init.R is None or best_init.t is None:
        raise RuntimeError(
            "Accepted initialization pair has no recovered pose."
        )

    if best_init.best_next_image is not None:
        next_support_text = (
            f"{best_init.best_next_correspondences} correspondences to image "
            f"{best_init.best_next_image}"
        )
    else:
        next_support_text = "no third-camera track support"

    log(
        f"init pair: ({image_0}, {image_1}) with {best_init.n_inliers} inliers, "
        f"parallax {best_init.median_parallax_deg:.1f}°, "
        f"expandable cameras {best_init.expandable_camera_count}, "
        f"best next support: {next_support_text}"
    )

    reconstruction.add_camera(
        image_0,
        CameraPose(
            R=np.eye(3),
            t=np.zeros(3),
        ),
    )

    reconstruction.add_camera(
        image_1,
        CameraPose(
            R=best_init.R,
            t=best_init.t,
        ),
    )

    initial_baseline = float(
        np.linalg.norm(
            reconstruction.camera_center(image_1)
            - reconstruction.camera_center(image_0)
        )
    )

    for evaluation in init_evaluations:
        diagnostics.record_pair_evaluation(evaluation.to_dict())

    state = ReconstructionState(
        reconstruction=reconstruction,
        track_manager=track_manager,
        keypoints=keypoints,
        config=config,
        diagnostics=diagnostics,
        registration_order=[
            image_0,
            image_1,
        ],
        anchor_camera=image_0,
        scale_camera=image_1,
        initial_baseline=initial_baseline,
    )

    n_new = triangulate_new_landmarks(
        state,
    )

    log(f"initial triangulation: {n_new} points")

    return InitializationResult(
        state=state,
        image_0=image_0,
        image_1=image_1,
    )
