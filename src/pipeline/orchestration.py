"""Incremental Structure-from-Motion pipeline orchestration."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..config import SfMConfig
from ..initialization import (
    InitialPairEvaluation,
    choose_best_initial_pair_evaluation,
    evaluate_pair,
    select_init_pair,
)
from ..models import (
    CameraIntrinsics,
    CameraPose,
    Reconstruction,
    ReconstructionState,
)
from ..tracks import TrackManager
from ..validation.diagnostics import (
    ReconstructionDiagnostics,
    ReconstructionStatus,
    TerminationReason,
)
from .diagnostics import _failure_stats, compute_view_graph_components
from .finalization import finalize_reconstruction
from .reconstruction_ops import triangulate_new_landmarks
from .registration import run_incremental_registration
from .verification import verify_match_graph


def run_sfm(
    keypoints: list[np.ndarray],
    match_graph: dict[
        tuple[int, int],
        np.ndarray,
    ],
    K: np.ndarray | CameraIntrinsics,
    config: SfMConfig | None = None,
    ba_every: int = 3,
    use_ba: bool = True,
    use_geometric_verification: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[
    Reconstruction,
    TrackManager,
    dict[str, Any],
]:
    """Run robust incremental SfM and return reconstruction, tracks, and stats."""
    if config is None:
        config = SfMConfig(
            ba_every_n_cameras=ba_every
        )

    intrinsics = (
        K
        if isinstance(
            K,
            CameraIntrinsics,
        )
        else CameraIntrinsics.from_matrix(
            K
        )
    )

    K_mat = intrinsics.K

    raw_pairs = len(
        match_graph
    )

    raw_matches = sum(
        len(matches)
        for matches
        in match_graph.values()
    )

    # ------------------------------------------------------------------
    # 1. Pairwise geometric verification.
    # ------------------------------------------------------------------

    if use_geometric_verification:
        (
            pair_geometries,
            verified_match_graph,
        ) = verify_match_graph(
            match_graph,
            keypoints,
            intrinsics,
            config,
            log,
        )

    else:
        log(
            "geometric verification "
            "disabled (ablation)"
        )

        pair_geometries = {}

        verified_match_graph = {
            pair: np.asarray(
                matches,
                dtype=np.int64,
            ).reshape(-1, 2)
            for pair, matches
            in match_graph.items()
        }

    # ------------------------------------------------------------------
    # 2. Conflict-aware feature tracks.
    # ------------------------------------------------------------------

    track_manager = TrackManager(
        (
            pair_geometries
            if use_geometric_verification
            else verified_match_graph
        ),
        keypoints=keypoints,
        K=K_mat,
    )

    log(
        f"tracks: "
        f"{len(track_manager.tracks)}"
    )

    diagnostics = (
        ReconstructionDiagnostics()
    )

    (
        diagnostics
        .track_conflict_rejected_edges
    ) = track_manager.n_rejected_edges

    reconstruction = Reconstruction(
        intrinsics=intrinsics
    )

    if (
        use_geometric_verification
        and not pair_geometries
    ):
        diagnostics.status = (
            ReconstructionStatus.FAILED
        )

        diagnostics.termination_reason = (
            TerminationReason
            .INSUFFICIENT_VERIFIED_GEOMETRY
        )

        diagnostics.view_graph = (
            compute_view_graph_components(
                len(keypoints),
                [],
                set(),
            )
        )

        stats = _failure_stats(
            reconstruction=reconstruction,
            track_manager=track_manager,
            diagnostics=diagnostics,
            n_images=len(keypoints),
            raw_pairs=raw_pairs,
            raw_matches=raw_matches,
            verified_pairs=0,
            verified_matches=0,
            use_ba=use_ba,
            use_geometric_verification=(
                use_geometric_verification
            ),
        )

        return (
            reconstruction,
            track_manager,
            stats,
        )

    # ------------------------------------------------------------------
    # 3. Two-view initialization.
    #
    # Geometric validity determines whether a pair is accepted.
    # Among accepted pairs, Track expansion support is used so an
    # isolated high-parallax pair does not trap incremental SfM.
    # ------------------------------------------------------------------

    init_evaluations: list[
        InitialPairEvaluation
    ] = []

    try:
        if use_geometric_verification:
            (
                best_init,
                init_evaluations,
            ) = select_init_pair(
                pair_geometries,
                keypoints,
                K_mat,
                config,
                tracks=(
                    track_manager.tracks
                ),
            )

        else:
            # Fallback / ablation path.
            #
            # Here Essential estimation is performed directly
            # from the raw pair matches.
            init_evaluations = [
                evaluate_pair(
                    image_a,
                    image_b,
                    matches,
                    keypoints,
                    K_mat,
                    config,
                )
                for (
                    image_a,
                    image_b,
                ), matches
                in match_graph.items()
            ]

            best_init = (
                choose_best_initial_pair_evaluation(
                    init_evaluations,
                    tracks=(
                        track_manager.tracks
                    ),
                    min_pnp_correspondences=(
                        config
                        .min_pnp_correspondences
                    ),
                )
            )

    except RuntimeError:
        # Initialization failed.
        #
        # select_init_pair() performs deterministic pair evaluation
        # from already-computed PairGeometry. Re-evaluating those
        # candidates here does not repeat Essential RANSAC; it simply
        # preserves detailed failure diagnostics.

        if (
            use_geometric_verification
            and pair_geometries
        ):
            from ..initialization import (
                evaluate_pair_geometry,
            )

            init_evaluations = [
                evaluate_pair_geometry(
                    geometry,
                    keypoints,
                    K_mat,
                    config,
                )
                for geometry
                in pair_geometries.values()
            ]

        for evaluation in (
            init_evaluations
        ):
            (
                diagnostics
                .record_pair_evaluation(
                    evaluation.to_dict()
                )
            )

        diagnostics.status = (
            ReconstructionStatus.FAILED
        )

        diagnostics.termination_reason = (
            TerminationReason
            .INITIALIZATION_FAILED
        )

        diagnostics.view_graph = (
            compute_view_graph_components(
                len(keypoints),
                list(
                    verified_match_graph
                    .keys()
                ),
                set(),
            )
        )

        stats = _failure_stats(
            reconstruction=reconstruction,
            track_manager=track_manager,
            diagnostics=diagnostics,
            n_images=len(keypoints),
            raw_pairs=raw_pairs,
            raw_matches=raw_matches,
            verified_pairs=len(
                verified_match_graph
            ),
            verified_matches=sum(
                len(matches)
                for matches
                in verified_match_graph.values()
            ),
            use_ba=use_ba,
            use_geometric_verification=(
                use_geometric_verification
            ),
        )

        return (
            reconstruction,
            track_manager,
            stats,
        )

    # Successful initialization reaches here.
    image_0 = (
        best_init.image_a
    )

    image_1 = (
        best_init.image_b
    )

    if (
        best_init.R is None
        or best_init.t is None
    ):
        raise RuntimeError(
            "Accepted initialization pair "
            "has no recovered pose."
        )

    if (
        best_init.best_next_image
        is not None
    ):
        next_support_text = (
            f"{best_init.best_next_correspondences} "
            f"correspondences to image "
            f"{best_init.best_next_image}"
        )
    else:
        next_support_text = (
            "no third-camera track support"
        )

    log(
        f"init pair: "
        f"({image_0}, {image_1}) with "
        f"{best_init.n_inliers} inliers, "
        f"parallax "
        f"{best_init.median_parallax_deg:.1f}°, "
        f"expandable cameras "
        f"{best_init.expandable_camera_count}, "
        f"best next support: "
        f"{next_support_text}"
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
            reconstruction.camera_center(
                image_1
            )
            - reconstruction.camera_center(
                image_0
            )
        )
    )

    for evaluation in (
        init_evaluations
    ):
        diagnostics.record_pair_evaluation(
            evaluation.to_dict()
        )

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
        initial_baseline=(
            initial_baseline
        ),
    )

    n_new = triangulate_new_landmarks(
        state,
        log,
    )

    log(
        f"initial triangulation: "
        f"{n_new} points"
    )

    # ------------------------------------------------------------------
    # 4. Incremental camera registration.
    # ------------------------------------------------------------------

    failed_pnp_images = run_incremental_registration(
        state,
        use_ba=use_ba,
        log=log,
    )

    # ------------------------------------------------------------------
    # 5. Finalization (final optimization, camera statuses, stats, summary).
    # ------------------------------------------------------------------

    finalize_reconstruction(
        state,
        failed_pnp_images=failed_pnp_images,
        pair_geometries=pair_geometries,
        match_graph=match_graph,
        verified_match_graph=verified_match_graph,
        image_0=image_0,
        image_1=image_1,
        raw_pairs=raw_pairs,
        raw_matches=raw_matches,
        use_ba=use_ba,
        use_geometric_verification=use_geometric_verification,
        log=log,
    )

    return (
        reconstruction,
        track_manager,
        state.stats,
    )
