"""Incremental camera selection and registration helpers."""

from __future__ import annotations

from ..models import ReconstructionState


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
