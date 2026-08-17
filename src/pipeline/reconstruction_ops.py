"""Reconstruction state operations: observations, RMS, triangulation, and pruning."""

from __future__ import annotations

import numpy as np

from ..geometry.projection import reproject
from ..models import ReconstructionState
from ..triangulation import triangulate_track


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

        for observation in track.observations.values():
            if observation.image_id not in state.registered_images:
                continue

            observations.append(
                (
                    observation.image_id,
                    track.point3d_id,
                    state.keypoints[observation.image_id][
                        observation.keypoint_id
                    ],
                )
            )

    return observations


def compute_reprojection_rms(
    state: ReconstructionState,
) -> float | None:
    """Compute current reconstruction RMS reprojection error."""
    observations = collect_bundle_observations(state)
    if not observations:
        return None

    reconstruction = state.reconstruction
    squared_errors: list[float] = []

    for image_id, landmark_id, measured_xy in observations:
        pose = reconstruction.get_pose(image_id)
        xyz = reconstruction.landmarks[landmark_id].xyz.reshape(1, 3)
        projected = reproject(
            reconstruction.K,
            pose.R,
            pose.t,
            xyz,
        )[0]
        squared_errors.append(
            float(np.sum((projected - measured_xy) ** 2))
        )

    return float(np.sqrt(np.mean(squared_errors)))


def triangulate_new_landmarks(
    state: ReconstructionState,
) -> int:
    """Triangulate every untriangulated track visible in >=2 registered views."""
    reconstruction = state.reconstruction
    track_manager = state.track_manager
    created = 0

    for track_id in track_manager.triangulatable_tracks(
        state.registered_images
    ):
        track = track_manager.tracks[track_id]

        result = triangulate_track(
            track,
            reconstruction.cameras,
            state.keypoints,
            reconstruction.intrinsics,
            state.config,
        )

        if result.accepted and result.xyz is not None:
            landmark_id = reconstruction.add_landmark(
                xyz=result.xyz,
                track_id=track.id,
                reprojection_error_px=result.max_reprojection_error_px,
            )
            track.point3d_id = landmark_id
            created += 1

        elif result.reason is not None:
            state.diagnostics.record_triangulation_rejection(
                result.reason
            )

    return created


def prune_outliers(
    state: ReconstructionState,
    max_err: float,
) -> int:
    """Remove landmarks whose mean registered-view reprojection error is high."""
    reconstruction = state.reconstruction
    tracks = state.track_manager.tracks

    if not reconstruction.landmarks:
        return 0

    keep = np.ones(
        len(reconstruction.landmarks),
        dtype=bool,
    )

    for track in tracks:
        if track.point3d_id is None:
            continue

        landmark = reconstruction.landmarks[track.point3d_id]
        errors: list[float] = []

        for observation in track.observations.values():
            image_id = observation.image_id
            if image_id not in state.registered_images:
                continue

            pose = reconstruction.get_pose(image_id)
            projected = reproject(
                reconstruction.K,
                pose.R,
                pose.t,
                landmark.xyz.reshape(1, 3),
            )[0]
            measured = state.keypoints[image_id][
                observation.keypoint_id
            ]
            errors.append(
                float(np.linalg.norm(projected - measured))
            )

        if errors:
            mean_error = float(np.mean(errors))
            landmark.reprojection_error_px = mean_error
            if mean_error > max_err:
                keep[landmark.id] = False

    n_removed = reconstruction.retain_landmarks(
        keep,
        tracks,
    )
    state.diagnostics.landmark_removals += n_removed
    return n_removed
