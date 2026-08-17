"""Post-registration finalization, optimization, camera status diagnosis, and stats."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..models import PairGeometry, ReconstructionState
from ..pnp import PnPResult
from ..validation.diagnostics import (
    CameraRegistrationStatus,
    FailureReason,
    ReconstructionStatus,
    TerminationReason,
)
from .bundle import run_bundle_adjustment_on_state
from .diagnostics import compute_view_graph_components
from .reconstruction_ops import (
    collect_bundle_observations,
    compute_reprojection_rms,
    prune_outliers,
)


def run_final_optimization(
    state: ReconstructionState,
    *,
    use_ba: bool = True,
    log: Callable[[str], None] = print,
) -> list[tuple[int, int, np.ndarray]]:
    """Run final bundle adjustment, outlier pruning, and observation gathering."""
    reconstruction = state.reconstruction
    config = state.config

    ba_history = state.stats.setdefault(
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
        ) = run_bundle_adjustment_on_state(
            state
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
                config.final_prune_error_px
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
            ) = run_bundle_adjustment_on_state(
                state
            )

            state.stats[
                "rms_final"
            ] = rms_final

            ba_history.append(
                {
                    "stage": "post-prune",
                    "n_registered": len(
                        state.registered_images
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
                    "n_pruned": n_pruned,
                }
            )

            log(
                f"post-prune BA: rms "
                f"{rms_before_pruned:.3f} -> "
                f"{rms_final:.3f} px"
            )

        else:
            state.stats["rms_final"] = None

    if not use_ba:
        state.stats["rms_before_final_ba"] = None
        state.stats["rms_final"] = compute_reprojection_rms(
            state
        )

    observations = collect_bundle_observations(state)
    return observations


def finalize_camera_statuses(
    state: ReconstructionState,
    *,
    failed_pnp_images: dict[int, PnPResult],
    verified_match_graph: dict[tuple[int, int], np.ndarray],
) -> dict[str, Any]:
    """Diagnose remaining cameras, view graph components, and termination reason."""
    diagnostics = state.diagnostics
    track_manager = state.track_manager
    reconstruction = state.reconstruction
    keypoints = state.keypoints
    config = state.config

    view_graph_info = compute_view_graph_components(
        len(keypoints),
        list(verified_match_graph.keys()),
        state.registered_images,
    )

    diagnostics.view_graph = view_graph_info

    for image_id in range(len(keypoints)):
        if image_id in state.registered_images:
            diagnostics.camera_status[str(image_id)] = {
                "status": CameraRegistrationStatus.REGISTERED.value,
                "registration_order": state.registration_order.index(image_id),
                "is_registered": True,
            }
            continue

        points2d, _, _ = track_manager.correspondences_2d3d(
            image_id,
            keypoints,
            reconstruction.points3d,
        )
        n_correspondences = len(points2d)

        if image_id in failed_pnp_images:
            result = failed_pnp_images[image_id]
            reason = result.reason

            diagnostics.camera_status[str(image_id)] = {
                "status": CameraRegistrationStatus.REJECTED_PNP.value,
                "last_2d3d_correspondences": n_correspondences,
                "pnp_attempted": True,
                "pnp_inliers": len(result.inlier_indices),
                "pnp_inlier_ratio": result.inlier_ratio,
                "pnp_reprojection_rms_px": result.reprojection_rms_px,
                "rejection_reason": (
                    reason.value
                    if isinstance(reason, FailureReason)
                    else reason
                ),
                "is_registered": False,
            }
        else:
            diagnostics.camera_status[str(image_id)] = {
                "status": (
                    CameraRegistrationStatus.BLOCKED_NO_2D3D_CORRESPONDENCES.value
                ),
                "last_2d3d_correspondences": n_correspondences,
                "pnp_attempted": False,
                "rejection_reason": (
                    FailureReason.INSUFFICIENT_PNP_CORRESPONDENCES.value
                ),
                "is_registered": False,
            }

    remaining_images = set(range(len(keypoints))) - state.registered_images

    if not remaining_images:
        diagnostics.status = ReconstructionStatus.SUCCESS
        diagnostics.termination_reason = TerminationReason.COMPLETED
    else:
        diagnostics.status = ReconstructionStatus.PARTIAL

        remaining_correspondences = {
            image_id: int(
                diagnostics.camera_status[str(image_id)].get(
                    "last_2d3d_correspondences",
                    0,
                )
            )
            for image_id in remaining_images
        }

        all_lack_support = all(
            count < config.min_pnp_correspondences
            for count in remaining_correspondences.values()
        )

        active_component = set(view_graph_info["active_component"])

        all_remaining_outside_active = (
            bool(remaining_images)
            and remaining_images.isdisjoint(active_component)
        )

        if (
            all_lack_support
            and view_graph_info["num_components"] > 1
            and all_remaining_outside_active
        ):
            diagnostics.termination_reason = TerminationReason.NO_2D3D_BRIDGE
        else:
            diagnostics.termination_reason = (
                TerminationReason.NO_REGISTRABLE_CAMERA
            )

    return view_graph_info


def build_final_stats(
    state: ReconstructionState,
    *,
    observations: list[tuple[int, int, np.ndarray]],
    pair_geometries: dict[tuple[int, int], PairGeometry],
    match_graph: dict[tuple[int, int], np.ndarray],
    verified_match_graph: dict[tuple[int, int], np.ndarray],
    image_0: int,
    image_1: int,
    raw_pairs: int,
    raw_matches: int,
    use_ba: bool,
    use_geometric_verification: bool,
) -> None:
    """Populate state.stats and diagnostics.summary_stats."""
    diagnostics = state.diagnostics
    reconstruction = state.reconstruction
    keypoints = state.keypoints
    ba_history = state.stats.get("ba_history", [])

    # Preserve the actual verified initialization matches for plotting.
    initial_geometry = pair_geometries.get((image_0, image_1))
    if initial_geometry is None:
        initial_geometry = pair_geometries.get((image_1, image_0))

    if initial_geometry is not None:
        initial_verified_matches = initial_geometry.verified_matches.copy()
    else:
        raw_initial_matches = match_graph.get((image_0, image_1))
        if raw_initial_matches is None:
            raw_initial_matches = match_graph.get(
                (image_1, image_0),
                np.zeros((0, 2), dtype=np.int64),
            )
        initial_verified_matches = np.asarray(
            raw_initial_matches,
            dtype=np.int64,
        ).reshape(-1, 2)

    state.stats.update(
        {
            "registration_order": state.registration_order,
            "n_registered": len(state.registered_images),
            "n_images": len(keypoints),
            "n_points": len(reconstruction.landmarks),
            "n_observations": len(observations),
            "mean_track_length": (
                len(observations)
                / max(
                    len(reconstruction.landmarks),
                    1,
                )
            ),
            "use_ba": use_ba,
            "use_geometric_verification": use_geometric_verification,
            "raw_match_pairs": raw_pairs,
            "raw_matches": raw_matches,
            "verified_match_pairs": len(verified_match_graph),
            "verified_matches": sum(
                len(matches) for matches in verified_match_graph.values()
            ),
            "ba_history": ba_history,
            "diagnostics": diagnostics,
            "status": diagnostics.status.value,
            "termination_reason": diagnostics.termination_reason.value,
            "initial_pair_verified_matches": initial_verified_matches,
        }
    )

    diagnostics.summary_stats = {
        "initial_pair": [
            image_0,
            image_1,
        ],
        "registered_images": state.stats["n_registered"],
        "total_images": state.stats["n_images"],
        "landmarks": state.stats["n_points"],
        "observations": state.stats["n_observations"],
        "mean_track_length": state.stats["mean_track_length"],
        "final_reprojection_rms_px": state.stats.get("rms_final"),
        "status": diagnostics.status.value,
        "termination_reason": diagnostics.termination_reason.value,
        "registration_order": state.registration_order,
    }


def log_final_summary(
    state: ReconstructionState,
    *,
    view_graph_info: dict[str, Any],
    log: Callable[[str], None] = print,
) -> None:
    """Print the human-readable summary block for reconstruction results."""
    diagnostics = state.diagnostics

    most_common_triangulation = (
        max(
            diagnostics.triangulation_rejections.items(),
            key=lambda item: item[1],
        )
        if diagnostics.triangulation_rejections
        else (
            "None",
            0,
        )
    )

    blocked_count = sum(
        1
        for status in diagnostics.camera_status.values()
        if status["status"]
        == CameraRegistrationStatus.BLOCKED_NO_2D3D_CORRESPONDENCES.value
    )

    rejected_pnp_count = sum(
        1
        for status in diagnostics.camera_status.values()
        if status["status"] == CameraRegistrationStatus.REJECTED_PNP.value
    )

    log("\n" + "=" * 54)
    log(f"Reconstruction status : {diagnostics.status.value}")
    log(f"Termination reason    : {diagnostics.termination_reason.value}")
    log(
        f"Registered cameras    : "
        f"{state.stats['n_registered']}/{state.stats['n_images']}"
    )
    log(f"3D landmarks          : {state.stats['n_points']}")

    if state.stats.get("rms_final") is None:
        log("Reprojection RMS      : n/a")
    else:
        log(f"Reprojection RMS      : {state.stats['rms_final']:.3f} px")

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
        f"(active: {view_graph_info['active_component']})"
    )

    if most_common_triangulation[1] > 0:
        log(
            f"Top triangulation rej : "
            f"{most_common_triangulation[0]} "
            f"({most_common_triangulation[1]} tracks)"
        )

    log("=" * 54 + "\n")


def finalize_reconstruction(
    state: ReconstructionState,
    *,
    failed_pnp_images: dict[int, PnPResult],
    pair_geometries: dict[tuple[int, int], PairGeometry],
    match_graph: dict[tuple[int, int], np.ndarray],
    verified_match_graph: dict[tuple[int, int], np.ndarray],
    image_0: int,
    image_1: int,
    raw_pairs: int,
    raw_matches: int,
    use_ba: bool = True,
    use_geometric_verification: bool = True,
    log: Callable[[str], None] = print,
) -> None:
    """Sequentially execute all post-registration finalization steps."""
    observations = run_final_optimization(
        state,
        use_ba=use_ba,
        log=log,
    )

    view_graph_info = finalize_camera_statuses(
        state,
        failed_pnp_images=failed_pnp_images,
        verified_match_graph=verified_match_graph,
    )

    build_final_stats(
        state,
        observations=observations,
        pair_geometries=pair_geometries,
        match_graph=match_graph,
        verified_match_graph=verified_match_graph,
        image_0=image_0,
        image_1=image_1,
        raw_pairs=raw_pairs,
        raw_matches=raw_matches,
        use_ba=use_ba,
        use_geometric_verification=use_geometric_verification,
    )

    log_final_summary(
        state,
        view_graph_info=view_graph_info,
        log=log,
    )
