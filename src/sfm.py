"""Incremental Structure-from-Motion orchestration.

The mathematical stages live in dedicated modules. This file owns sequencing,
validation decisions, diagnostics, and synchronization between the canonical
typed reconstruction state and numerical solvers.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Callable

import numpy as np

from .bundle_adjustment import run_bundle_adjustment
from .config import SfMConfig
from .geometry.essential import ransac_essential, sampson_distance
from .geometry.projection import reproject, to_normalized
from .initialization import (
    InitialPairEvaluation,
    choose_best_initial_pair_evaluation,
    evaluate_pair,
    select_init_pair,
)
from .models import (
    CameraIntrinsics,
    CameraPose,
    PairGeometry,
    Reconstruction,
    ReconstructionState,
)
from .pnp import PnPResult, estimate_camera_pose
from .tracks import TrackManager
from .triangulation import triangulate_track
from .validation.diagnostics import (
    CameraRegistrationStatus,
    DiagnosticEvent,
    FailureReason,
    ReconstructionDiagnostics,
    ReconstructionStatus,
    TerminationReason,
)


def _sampson_thresh(
    K: np.ndarray,
    px: float = 3.0,
) -> float:
    """Convert an approximate pixel threshold to normalized Sampson units."""

    f = float(
        (K[0, 0] + K[1, 1]) / 2.0
    )
    return (px / f) ** 2


def compute_view_graph_components(
    num_images: int,
    verified_pairs: list[tuple[int, int]],
    registered_images: set[int],
) -> dict[str, Any]:
    """Compute connected components of the verified two-view graph."""

    adjacency: dict[int, set[int]] = {
        i: set()
        for i in range(num_images)
    }

    for image_a, image_b in verified_pairs:
        adjacency[image_a].add(image_b)
        adjacency[image_b].add(image_a)

    visited: set[int] = set()
    components: list[list[int]] = []

    for image_id in range(num_images):
        if image_id in visited:
            continue

        component: list[int] = []
        queue = deque([image_id])
        visited.add(image_id)

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in sorted(
                adjacency[current]
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(
            sorted(component)
        )

    components.sort(
        key=lambda component: (
            -len(component),
            component,
        )
    )

    active_component: list[int] = []
    other_components: list[list[int]] = []
    isolated_images: list[int] = []

    for component in components:
        if (
            len(component) == 1
            and not adjacency[component[0]]
        ):
            isolated_images.append(
                component[0]
            )

        elif any(
            image_id in registered_images
            for image_id in component
        ):
            active_component.extend(
                component
            )

        else:
            other_components.append(
                component
            )

    return {
        "num_components": len(components),
        "components": components,
        "active_component": sorted(
            set(active_component)
        ),
        "other_components": other_components,
        "isolated_images": sorted(
            isolated_images
        ),
    }


def verify_match_graph(
    match_graph: dict[
        tuple[int, int],
        np.ndarray,
    ],
    keypoints: list[np.ndarray],
    intrinsics: CameraIntrinsics | np.ndarray,
    config: SfMConfig,
    log: Callable[[str], None] = print,
) -> tuple[
    dict[tuple[int, int], PairGeometry],
    dict[tuple[int, int], np.ndarray],
]:
    """Estimate Essential geometry exactly once for each candidate pair."""

    K = (
        intrinsics.K
        if isinstance(
            intrinsics,
            CameraIntrinsics,
        )
        else np.asarray(
            intrinsics,
            dtype=float,
        )
    )

    pair_geometries: dict[
        tuple[int, int],
        PairGeometry,
    ] = {}

    verified_matches: dict[
        tuple[int, int],
        np.ndarray,
    ] = {}

    threshold = _sampson_thresh(
        K,
        px=config.essential_ransac_px,
    )

    for (
        image_a,
        image_b,
    ), matches_in in sorted(
        match_graph.items()
    ):
        matches = np.asarray(
            matches_in,
            dtype=np.int64,
        ).reshape(-1, 2)

        if (
            len(matches)
            < config.min_verified_matches
        ):
            continue

        x_a = to_normalized(
            keypoints[image_a][
                matches[:, 0]
            ],
            K,
        )

        x_b = to_normalized(
            keypoints[image_b][
                matches[:, 1]
            ],
            K,
        )

        pair_seed = int(
            (
                image_a * 10007
                + image_b * 997
                + config.global_seed
            )
            % (2**31 - 1)
        )

        try:
            E, inlier_mask = (
                ransac_essential(
                    x_a,
                    x_b,
                    thresh=threshold,
                    iters=(
                        config
                        .essential_ransac_iters
                    ),
                    seed=pair_seed,
                )
            )

        except RuntimeError:
            continue

        n_inliers = int(
            inlier_mask.sum()
        )

        if (
            n_inliers
            < config.min_verified_matches
        ):
            continue

        pair_matches = matches[
            inlier_mask
        ]

        errors = sampson_distance(
            E,
            x_a[inlier_mask],
            x_b[inlier_mask],
        )

        geometry = PairGeometry(
            image_a=image_a,
            image_b=image_b,
            raw_matches=matches,
            verified_matches=pair_matches,
            E=E,
            inlier_mask=inlier_mask,
            n_raw_matches=len(matches),
            n_inliers=n_inliers,
            inlier_ratio=float(
                n_inliers
                / len(matches)
            ),
            median_sampson_error=(
                float(
                    np.median(errors)
                )
                if len(errors)
                else None
            ),
        )

        pair_geometries[
            (image_a, image_b)
        ] = geometry

        verified_matches[
            (image_a, image_b)
        ] = pair_matches

    n_raw = sum(
        len(matches)
        for matches
        in match_graph.values()
    )

    n_verified = sum(
        len(matches)
        for matches
        in verified_matches.values()
    )

    log(
        f"geometric verification: "
        f"{n_raw} -> {n_verified} matches, "
        f"{len(match_graph)} -> "
        f"{len(verified_matches)} pairs"
    )

    return (
        pair_geometries,
        verified_matches,
    )


def collect_bundle_observations(
    state: ReconstructionState,
) -> list[
    tuple[
        int,
        int,
        np.ndarray,
    ]
]:
    """Return (image_id, landmark_id, xy_pixel) observations for BA."""

    observations: list[
        tuple[
            int,
            int,
            np.ndarray,
        ]
    ] = []

    for track in state.track_manager.tracks:
        if track.point3d_id is None:
            continue

        for observation in (
            track.observations.values()
        ):
            if (
                observation.image_id
                not in state.registered_images
            ):
                continue

            observations.append(
                (
                    observation.image_id,
                    track.point3d_id,
                    state.keypoints[
                        observation.image_id
                    ][
                        observation.keypoint_id
                    ],
                )
            )

    return observations


def compute_reprojection_rms(
    state: ReconstructionState,
) -> float | None:
    """Compute current reconstruction RMS reprojection error."""

    observations = (
        collect_bundle_observations(
            state
        )
    )

    if not observations:
        return None

    reconstruction = (
        state.reconstruction
    )

    squared_errors: list[
        float
    ] = []

    for (
        image_id,
        landmark_id,
        measured_xy,
    ) in observations:
        pose = reconstruction.get_pose(
            image_id
        )

        xyz = (
            reconstruction
            .landmarks[landmark_id]
            .xyz
            .reshape(1, 3)
        )

        projected = reproject(
            reconstruction.K,
            pose.R,
            pose.t,
            xyz,
        )[0]

        squared_errors.append(
            float(
                np.sum(
                    (
                        projected
                        - measured_xy
                    )
                    ** 2
                )
            )
        )

    return float(
        np.sqrt(
            np.mean(
                squared_errors
            )
        )
    )


def triangulate_new_landmarks(
    state: ReconstructionState,
    log: Callable[[str], None] | None = None,
) -> int:
    """Triangulate every untriangulated track visible in >=2 registered views."""

    reconstruction = (
        state.reconstruction
    )

    track_manager = (
        state.track_manager
    )

    created = 0

    for track_id in (
        track_manager
        .triangulatable_tracks(
            state.registered_images
        )
    ):
        track = (
            track_manager
            .tracks[track_id]
        )

        result = triangulate_track(
            track,
            reconstruction.cameras,
            state.keypoints,
            reconstruction.intrinsics,
            state.config,
        )

        if (
            result.accepted
            and result.xyz is not None
        ):
            landmark_id = (
                reconstruction
                .add_landmark(
                    xyz=result.xyz,
                    track_id=track.id,
                    reprojection_error_px=(
                        result
                        .max_reprojection_error_px
                    ),
                )
            )

            track.point3d_id = (
                landmark_id
            )

            created += 1

        elif result.reason is not None:
            (
                state.diagnostics
                .record_triangulation_rejection(
                    result.reason
                )
            )

    return created


def prune_outliers(
    state: ReconstructionState,
    max_err: float,
) -> int:
    """Remove landmarks whose mean registered-view reprojection error is high."""

    reconstruction = (
        state.reconstruction
    )

    tracks = (
        state.track_manager.tracks
    )

    if not reconstruction.landmarks:
        return 0

    keep = np.ones(
        len(
            reconstruction.landmarks
        ),
        dtype=bool,
    )

    for track in tracks:
        if track.point3d_id is None:
            continue

        landmark = (
            reconstruction
            .landmarks[
                track.point3d_id
            ]
        )

        errors: list[float] = []

        for observation in (
            track.observations.values()
        ):
            image_id = (
                observation.image_id
            )

            if (
                image_id
                not in state.registered_images
            ):
                continue

            pose = (
                reconstruction
                .get_pose(image_id)
            )

            projected = reproject(
                reconstruction.K,
                pose.R,
                pose.t,
                landmark.xyz.reshape(
                    1,
                    3,
                ),
            )[0]

            measured = (
                state.keypoints[
                    image_id
                ][
                    observation.keypoint_id
                ]
            )

            errors.append(
                float(
                    np.linalg.norm(
                        projected
                        - measured
                    )
                )
            )

        if errors:
            mean_error = float(
                np.mean(errors)
            )

            (
                landmark
                .reprojection_error_px
            ) = mean_error

            if mean_error > max_err:
                keep[
                    landmark.id
                ] = False

    n_removed = (
        reconstruction
        .retain_landmarks(
            keep,
            tracks,
        )
    )

    (
        state.diagnostics
        .landmark_removals
    ) += n_removed

    return n_removed


def choose_next_camera(
    state: ReconstructionState,
    deferred: set[int],
) -> tuple[
    int | None,
    int,
]:
    """Choose the unregistered image with the strongest current 2D-3D support."""

    best_image: int | None = None
    best_count = 0

    points3d = (
        state.reconstruction
        .points3d
    )

    for image_id in range(
        state.n_images
    ):
        if (
            image_id
            in state.registered_images
            or image_id in deferred
        ):
            continue

        points2d, _, _ = (
            state.track_manager
            .correspondences_2d3d(
                image_id,
                state.keypoints,
                points3d,
            )
        )

        if len(points2d) > best_count:
            best_image = image_id
            best_count = len(
                points2d
            )

    return (
        best_image,
        best_count,
    )


def _run_bundle_adjustment_on_state(
    state: ReconstructionState,
) -> tuple[
    float,
    float,
    int,
]:
    """Numerical BA boundary: typed state -> arrays -> typed state."""

    observations = (
        collect_bundle_observations(
            state
        )
    )

    reconstruction = (
        state.reconstruction
    )

    (
        poses,
        points,
        rms_before,
        rms_after,
    ) = run_bundle_adjustment(
        reconstruction.intrinsics,
        reconstruction.poses,
        reconstruction.points3d,
        observations,
        anchor_camera=(
            state.anchor_camera
        ),
        scale_camera=(
            state.scale_camera
        ),
        initial_baseline=(
            state.initial_baseline
        ),
        config=state.config,
    )

    reconstruction.apply_optimization(
        poses,
        points,
    )

    return (
        rms_before,
        rms_after,
        len(observations),
    )


def _failure_stats(
    *,
    reconstruction: Reconstruction,
    track_manager: TrackManager,
    diagnostics: ReconstructionDiagnostics,
    n_images: int,
    raw_pairs: int,
    raw_matches: int,
    verified_pairs: int,
    verified_matches: int,
    use_ba: bool,
    use_geometric_verification: bool,
) -> dict[str, Any]:
    """Build stats for a reconstruction that failed before initialization."""

    diagnostics.summary_stats = {
        "initial_pair": [],
        "registered_images": 0,
        "total_images": n_images,
        "landmarks": 0,
        "observations": 0,
        "mean_track_length": 0.0,
        "final_reprojection_rms_px": None,
        "status": diagnostics.status.value,
        "termination_reason": (
            diagnostics
            .termination_reason
            .value
        ),
        "registration_order": [],
    }

    return {
        "registration_order": [],
        "n_registered": 0,
        "n_images": n_images,
        "n_points": 0,
        "n_observations": 0,
        "mean_track_length": 0.0,
        "rms_before_final_ba": None,
        "rms_final": None,
        "use_ba": use_ba,
        "use_geometric_verification": (
            use_geometric_verification
        ),
        "raw_match_pairs": raw_pairs,
        "raw_matches": raw_matches,
        "verified_match_pairs": (
            verified_pairs
        ),
        "verified_matches": (
            verified_matches
        ),
        "ba_history": [],
        "diagnostics": diagnostics,
        "status": (
            diagnostics.status.value
        ),
        "termination_reason": (
            diagnostics
            .termination_reason
            .value
        ),
        "initial_pair_verified_matches": (
            np.zeros(
                (0, 2),
                dtype=np.int64,
            )
        ),
    }


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
            from .initialization import (
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

    since_ba = 0

    deferred: set[int] = set()

    failed_pnp_images: dict[
        int,
        PnPResult,
    ] = {}

    # ------------------------------------------------------------------
    # 4. Incremental camera registration.
    # ------------------------------------------------------------------

    while (
        state.has_unregistered_images()
    ):
        (
            best_image,
            n_correspondences,
        ) = choose_next_camera(
            state,
            deferred,
        )

        if (
            best_image is None
            or n_correspondences
            < config.min_pnp_correspondences
        ):
            break

        (
            points2d,
            points3d,
            _,
        ) = (
            track_manager
            .correspondences_2d3d(
                best_image,
                keypoints,
                reconstruction.points3d,
            )
        )

        state.pnp_attempt_count += 1

        pnp_result = (
            estimate_camera_pose(
                points3d,
                points2d,
                reconstruction.intrinsics,
                config,
            )
        )

        if (
            not pnp_result.accepted
            or pnp_result.pose is None
        ):
            state.pnp_rejected_count += 1

            reason = (
                pnp_result.reason
                or FailureReason.PNP_FAILED
            )

            diagnostics.record_pnp_rejection(
                reason
            )

            failed_pnp_images[
                best_image
            ] = pnp_result

            deferred.add(
                best_image
            )

            diagnostics.record(
                DiagnosticEvent(
                    stage=(
                        "CAMERA_REGISTRATION"
                    ),
                    entity=(
                        f"image:{best_image}"
                    ),
                    status="DEFERRED",
                    reason=reason,
                    metrics={
                        "correspondences": (
                            len(points2d)
                        ),
                        "inliers": len(
                            pnp_result
                            .inlier_indices
                        ),
                        "inlier_ratio": (
                            pnp_result
                            .inlier_ratio
                        ),
                        "reprojection_rms_px": (
                            pnp_result
                            .reprojection_rms_px
                        ),
                    },
                )
            )

            log(
                f"image {best_image}: "
                f"PnP rejected "
                f"({len(pnp_result.inlier_indices)}/"
                f"{len(points2d)} inliers, "
                f"reason: {reason}); "
                f"deferring"
            )

            continue

        state.pnp_accepted_count += 1

        state.add_camera(
            best_image,
            pnp_result.pose,
        )

        # The map has expanded. Previously deferred cameras may
        # now have enough 2D-3D support.
        deferred.clear()

        failed_pnp_images.pop(
            best_image,
            None,
        )

        diagnostics.record(
            DiagnosticEvent(
                stage=(
                    "CAMERA_REGISTRATION"
                ),
                entity=(
                    f"image:{best_image}"
                ),
                status="REGISTERED",
                metrics={
                    "correspondences": (
                        len(points2d)
                    ),
                    "inliers": len(
                        pnp_result
                        .inlier_indices
                    ),
                    "inlier_ratio": (
                        pnp_result
                        .inlier_ratio
                    ),
                    "reprojection_rms_px": (
                        pnp_result
                        .reprojection_rms_px
                    ),
                },
            )
        )

        n_new = (
            triangulate_new_landmarks(
                state,
                log,
            )
        )

        log(
            f"registered image "
            f"{best_image}: "
            f"{len(pnp_result.inlier_indices)}/"
            f"{len(points2d)} PnP inliers, "
            f"+{n_new} points "
            f"(total "
            f"{len(reconstruction.landmarks)})"
        )

        since_ba += 1

        if (
            use_ba
            and since_ba
            >= config.ba_every_n_cameras
        ):
            (
                rms_before,
                rms_after,
                n_observations,
            ) = (
                _run_bundle_adjustment_on_state(
                    state
                )
            )

            n_pruned = prune_outliers(
                state,
                max_err=(
                    config
                    .post_ba_prune_error_px
                ),
            )

            log(
                f"  BA: rms "
                f"{rms_before:.3f} -> "
                f"{rms_after:.3f} px "
                f"({n_observations} obs), "
                f"pruned {n_pruned}"
            )

            (
                state.stats
                .setdefault(
                    "ba_history",
                    [],
                )
                .append(
                    {
                        "stage": "periodic",
                        "n_registered": len(
                            state
                            .registered_images
                        ),
                        "n_observations": (
                            n_observations
                        ),
                        "rms_before_px": (
                            rms_before
                        ),
                        "rms_after_px": (
                            rms_after
                        ),
                        "n_pruned": (
                            n_pruned
                        ),
                    }
                )
            )

            since_ba = 0

    # ------------------------------------------------------------------
    # 5. Final BA and pruning.
    # ------------------------------------------------------------------

    ba_history = state.stats.get(
        "ba_history",
        [],
    )

    if (
        use_ba
        and reconstruction.landmarks
    ):
        (
            rms_before,
            rms_after,
            n_observations,
        ) = (
            _run_bundle_adjustment_on_state(
                state
            )
        )

        state.stats[
            "rms_before_final_ba"
        ] = rms_before

        ba_history.append(
            {
                "stage": "final",
                "n_registered": len(
                    state.registered_images
                ),
                "n_observations": (
                    n_observations
                ),
                "rms_before_px": (
                    rms_before
                ),
                "rms_after_px": (
                    rms_after
                ),
                "n_pruned": 0,
            }
        )

        log(
            f"final BA: rms "
            f"{rms_before:.3f} -> "
            f"{rms_after:.3f} px"
        )

        n_pruned = prune_outliers(
            state,
            max_err=(
                config
                .final_prune_error_px
            ),
        )

        log(
            f"pruned {n_pruned} "
            f"outlier points"
        )

        if reconstruction.landmarks:
            (
                rms_before_pruned,
                rms_final,
                n_observations,
            ) = (
                _run_bundle_adjustment_on_state(
                    state
                )
            )

            state.stats[
                "rms_final"
            ] = rms_final

            ba_history.append(
                {
                    "stage": (
                        "post-prune"
                    ),
                    "n_registered": len(
                        state
                        .registered_images
                    ),
                    "n_observations": (
                        n_observations
                    ),
                    "rms_before_px": (
                        rms_before_pruned
                    ),
                    "rms_after_px": (
                        rms_final
                    ),
                    "n_pruned": (
                        n_pruned
                    ),
                }
            )

            log(
                f"post-prune BA: rms "
                f"{rms_before_pruned:.3f} -> "
                f"{rms_final:.3f} px"
            )

        else:
            state.stats[
                "rms_final"
            ] = None

    if not use_ba:
        state.stats[
            "rms_before_final_ba"
        ] = None

        state.stats[
            "rms_final"
        ] = compute_reprojection_rms(
            state
        )

    observations = (
        collect_bundle_observations(
            state
        )
    )

    # ------------------------------------------------------------------
    # 6. Final camera statuses and termination diagnosis.
    # ------------------------------------------------------------------

    view_graph_info = (
        compute_view_graph_components(
            len(keypoints),
            list(
                verified_match_graph.keys()
            ),
            state.registered_images,
        )
    )

    diagnostics.view_graph = (
        view_graph_info
    )

    for image_id in range(
        len(keypoints)
    ):
        if (
            image_id
            in state.registered_images
        ):
            diagnostics.camera_status[
                str(image_id)
            ] = {
                "status": (
                    CameraRegistrationStatus
                    .REGISTERED
                    .value
                ),
                "registration_order": (
                    state
                    .registration_order
                    .index(image_id)
                ),
                "is_registered": True,
            }

            continue

        points2d, _, _ = (
            track_manager
            .correspondences_2d3d(
                image_id,
                keypoints,
                reconstruction.points3d,
            )
        )

        n_correspondences = len(
            points2d
        )

        if (
            image_id
            in failed_pnp_images
        ):
            result = (
                failed_pnp_images[
                    image_id
                ]
            )

            reason = result.reason

            diagnostics.camera_status[
                str(image_id)
            ] = {
                "status": (
                    CameraRegistrationStatus
                    .REJECTED_PNP
                    .value
                ),
                "last_2d3d_correspondences": (
                    n_correspondences
                ),
                "pnp_attempted": True,
                "pnp_inliers": len(
                    result.inlier_indices
                ),
                "pnp_inlier_ratio": (
                    result.inlier_ratio
                ),
                "pnp_reprojection_rms_px": (
                    result
                    .reprojection_rms_px
                ),
                "rejection_reason": (
                    reason.value
                    if isinstance(
                        reason,
                        FailureReason,
                    )
                    else reason
                ),
                "is_registered": False,
            }

        else:
            diagnostics.camera_status[
                str(image_id)
            ] = {
                "status": (
                    CameraRegistrationStatus
                    .BLOCKED_NO_2D3D_CORRESPONDENCES
                    .value
                ),
                "last_2d3d_correspondences": (
                    n_correspondences
                ),
                "pnp_attempted": False,
                "rejection_reason": (
                    FailureReason
                    .INSUFFICIENT_PNP_CORRESPONDENCES
                    .value
                ),
                "is_registered": False,
            }

    remaining_images = (
        set(
            range(
                len(keypoints)
            )
        )
        - state.registered_images
    )

    if not remaining_images:
        diagnostics.status = (
            ReconstructionStatus.SUCCESS
        )

        diagnostics.termination_reason = (
            TerminationReason.COMPLETED
        )

    else:
        diagnostics.status = (
            ReconstructionStatus.PARTIAL
        )

        remaining_correspondences = {
            image_id: int(
                diagnostics
                .camera_status[
                    str(image_id)
                ]
                .get(
                    "last_2d3d_correspondences",
                    0,
                )
            )
            for image_id
            in remaining_images
        }

        all_lack_support = all(
            count
            < config.min_pnp_correspondences
            for count
            in remaining_correspondences.values()
        )

        active_component = set(
            view_graph_info[
                "active_component"
            ]
        )

        # NO_2D3D_BRIDGE is valid only when the remaining images
        # are genuinely outside the active verified component.
        all_remaining_outside_active = (
            bool(remaining_images)
            and remaining_images.isdisjoint(
                active_component
            )
        )

        if (
            all_lack_support
            and view_graph_info[
                "num_components"
            ] > 1
            and all_remaining_outside_active
        ):
            diagnostics.termination_reason = (
                TerminationReason
                .NO_2D3D_BRIDGE
            )

        else:
            diagnostics.termination_reason = (
                TerminationReason
                .NO_REGISTRABLE_CAMERA
            )

    # ------------------------------------------------------------------
    # Preserve the actual verified initialization matches for plotting.
    # ------------------------------------------------------------------

    initial_geometry = (
        pair_geometries.get(
            (image_0, image_1)
        )
    )

    if initial_geometry is None:
        initial_geometry = (
            pair_geometries.get(
                (image_1, image_0)
            )
        )

    if initial_geometry is not None:
        initial_verified_matches = (
            initial_geometry
            .verified_matches
            .copy()
        )

    else:
        raw_initial_matches = (
            match_graph.get(
                (image_0, image_1)
            )
        )

        if raw_initial_matches is None:
            raw_initial_matches = (
                match_graph.get(
                    (image_1, image_0),
                    np.zeros(
                        (0, 2),
                        dtype=np.int64,
                    ),
                )
            )

        initial_verified_matches = (
            np.asarray(
                raw_initial_matches,
                dtype=np.int64,
            )
            .reshape(-1, 2)
        )

    state.stats.update(
        {
            "registration_order": (
                state.registration_order
            ),
            "n_registered": len(
                state.registered_images
            ),
            "n_images": len(
                keypoints
            ),
            "n_points": len(
                reconstruction.landmarks
            ),
            "n_observations": len(
                observations
            ),
            "mean_track_length": (
                len(observations)
                / max(
                    len(
                        reconstruction
                        .landmarks
                    ),
                    1,
                )
            ),
            "use_ba": use_ba,
            "use_geometric_verification": (
                use_geometric_verification
            ),
            "raw_match_pairs": (
                raw_pairs
            ),
            "raw_matches": (
                raw_matches
            ),
            "verified_match_pairs": len(
                verified_match_graph
            ),
            "verified_matches": sum(
                len(matches)
                for matches
                in verified_match_graph.values()
            ),
            "ba_history": (
                ba_history
            ),
            "diagnostics": (
                diagnostics
            ),
            "status": (
                diagnostics.status.value
            ),
            "termination_reason": (
                diagnostics
                .termination_reason
                .value
            ),
            "initial_pair_verified_matches": (
                initial_verified_matches
            ),
        }
    )

    diagnostics.summary_stats = {
        "initial_pair": [
            image_0,
            image_1,
        ],
        "registered_images": (
            state.stats[
                "n_registered"
            ]
        ),
        "total_images": (
            state.stats[
                "n_images"
            ]
        ),
        "landmarks": (
            state.stats[
                "n_points"
            ]
        ),
        "observations": (
            state.stats[
                "n_observations"
            ]
        ),
        "mean_track_length": (
            state.stats[
                "mean_track_length"
            ]
        ),
        "final_reprojection_rms_px": (
            state.stats.get(
                "rms_final"
            )
        ),
        "status": (
            diagnostics.status.value
        ),
        "termination_reason": (
            diagnostics
            .termination_reason
            .value
        ),
        "registration_order": (
            state.registration_order
        ),
    }

    most_common_triangulation = (
        max(
            diagnostics
            .triangulation_rejections
            .items(),
            key=lambda item: (
                item[1]
            ),
        )
        if (
            diagnostics
            .triangulation_rejections
        )
        else (
            "None",
            0,
        )
    )

    blocked_count = sum(
        1
        for status
        in diagnostics
        .camera_status
        .values()
        if (
            status["status"]
            == (
                CameraRegistrationStatus
                .BLOCKED_NO_2D3D_CORRESPONDENCES
                .value
            )
        )
    )

    rejected_pnp_count = sum(
        1
        for status
        in diagnostics
        .camera_status
        .values()
        if (
            status["status"]
            == (
                CameraRegistrationStatus
                .REJECTED_PNP
                .value
            )
        )
    )

    log(
        "\n"
        + "=" * 54
    )

    log(
        f"Reconstruction status : "
        f"{diagnostics.status.value}"
    )

    log(
        f"Termination reason    : "
        f"{diagnostics.termination_reason.value}"
    )

    log(
        f"Registered cameras    : "
        f"{state.stats['n_registered']}/"
        f"{state.stats['n_images']}"
    )

    log(
        f"3D landmarks          : "
        f"{state.stats['n_points']}"
    )

    if (
        state.stats.get(
            "rms_final"
        )
        is None
    ):
        log(
            "Reprojection RMS      : n/a"
        )

    else:
        log(
            f"Reprojection RMS      : "
            f"{state.stats['rms_final']:.3f} px"
        )

    log(
        f"PnP cameras           : "
        f"{state.pnp_accepted_count} accepted, "
        f"{state.pnp_rejected_count} rejected "
        f"({state.pnp_attempt_count} attempts)"
    )

    log(
        f"Remaining cameras     : "
        f"{blocked_count} blocked by 2D<->3D support, "
        f"{rejected_pnp_count} rejected by PnP"
    )

    log(
        f"Verified view graph   : "
        f"{view_graph_info['num_components']} components "
        f"(active: "
        f"{view_graph_info['active_component']})"
    )

    if (
        most_common_triangulation[1]
        > 0
    ):
        log(
            f"Top triangulation rej : "
            f"{most_common_triangulation[0]} "
            f"({most_common_triangulation[1]} tracks)"
        )

    log(
        "=" * 54
        + "\n"
    )

    return (
        reconstruction,
        track_manager,
        state.stats,
    )