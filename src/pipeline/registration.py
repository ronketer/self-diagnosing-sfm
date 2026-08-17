"""Incremental camera selection and registration helpers."""

from __future__ import annotations

from typing import Callable

from ..models import ReconstructionState
from ..pnp import PnPResult, estimate_camera_pose
from ..validation.diagnostics import (
    DiagnosticEvent,
    FailureReason,
)
from .bundle import _run_bundle_adjustment_on_state
from .reconstruction_ops import (
    prune_outliers,
    triangulate_new_landmarks,
)


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
    points3d = state.reconstruction.points3d

    for image_id in range(state.n_images):
        if (
            image_id in state.registered_images
            or image_id in deferred
        ):
            continue

        points2d, _, _ = (
            state.track_manager.correspondences_2d3d(
                image_id,
                state.keypoints,
                points3d,
            )
        )

        if len(points2d) > best_count:
            best_image = image_id
            best_count = len(points2d)

    return (
        best_image,
        best_count,
    )


def run_incremental_registration(
    state: ReconstructionState,
    *,
    use_ba: bool = True,
    log: Callable[[str], None] = print,
) -> dict[int, PnPResult]:
    """Register remaining cameras incrementally via PnP and periodic BA."""
    reconstruction = state.reconstruction
    track_manager = state.track_manager
    keypoints = state.keypoints
    config = state.config
    diagnostics = state.diagnostics

    since_ba = 0
    deferred: set[int] = set()
    failed_pnp_images: dict[int, PnPResult] = {}

    while state.has_unregistered_images():
        best_image, n_correspondences = choose_next_camera(
            state,
            deferred,
        )

        if (
            best_image is None
            or n_correspondences < config.min_pnp_correspondences
        ):
            break

        points2d, points3d, _ = track_manager.correspondences_2d3d(
            best_image,
            keypoints,
            reconstruction.points3d,
        )

        state.pnp_attempt_count += 1

        pnp_result = estimate_camera_pose(
            points3d,
            points2d,
            reconstruction.intrinsics,
            config,
        )

        if not pnp_result.accepted or pnp_result.pose is None:
            state.pnp_rejected_count += 1
            reason = pnp_result.reason or FailureReason.PNP_FAILED

            diagnostics.record_pnp_rejection(reason)
            failed_pnp_images[best_image] = pnp_result
            deferred.add(best_image)

            diagnostics.record(
                DiagnosticEvent(
                    stage="CAMERA_REGISTRATION",
                    entity=f"image:{best_image}",
                    status="DEFERRED",
                    reason=reason,
                    metrics={
                        "correspondences": len(points2d),
                        "inliers": len(pnp_result.inlier_indices),
                        "inlier_ratio": pnp_result.inlier_ratio,
                        "reprojection_rms_px": pnp_result.reprojection_rms_px,
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
        state.add_camera(best_image, pnp_result.pose)

        # The map has expanded. Previously deferred cameras may now have enough 2D-3D support.
        deferred.clear()
        failed_pnp_images.pop(best_image, None)

        diagnostics.record(
            DiagnosticEvent(
                stage="CAMERA_REGISTRATION",
                entity=f"image:{best_image}",
                status="REGISTERED",
                metrics={
                    "correspondences": len(points2d),
                    "inliers": len(pnp_result.inlier_indices),
                    "inlier_ratio": pnp_result.inlier_ratio,
                    "reprojection_rms_px": pnp_result.reprojection_rms_px,
                },
            )
        )

        n_new = triangulate_new_landmarks(state, log)

        log(
            f"registered image {best_image}: "
            f"{len(pnp_result.inlier_indices)}/{len(points2d)} PnP inliers, "
            f"+{n_new} points (total {len(reconstruction.landmarks)})"
        )

        since_ba += 1

        if use_ba and since_ba >= config.ba_every_n_cameras:
            (
                rms_before,
                rms_after,
                n_observations,
            ) = _run_bundle_adjustment_on_state(state)

            n_pruned = prune_outliers(
                state,
                max_err=config.post_ba_prune_error_px,
            )

            log(
                f"  BA: rms {rms_before:.3f} -> {rms_after:.3f} px "
                f"({n_observations} obs), pruned {n_pruned}"
            )

            state.stats.setdefault("ba_history", []).append(
                {
                    "stage": "periodic",
                    "n_registered": len(state.registered_images),
                    "n_observations": n_observations,
                    "rms_before_px": rms_before,
                    "rms_after_px": rms_after,
                    "n_pruned": n_pruned,
                }
            )

            since_ba = 0

    return failed_pnp_images
