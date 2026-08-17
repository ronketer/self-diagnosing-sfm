"""Unit tests for canonical reconstruction state and typed domain models."""

import numpy as np

from src.models import (
    CameraIntrinsics,
    CameraPose,
    Observation,
    Reconstruction,
    Track,
)


def test_reconstruction_camera_and_landmark_state():
    K = np.array(
        [
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )

    reconstruction = Reconstruction(
        intrinsics=CameraIntrinsics.from_matrix(K)
    )

    pose = CameraPose(
        R=np.eye(3),
        t=np.array([1.0, 0.0, 0.0]),
    )
    reconstruction.add_camera(3, pose)

    assert set(reconstruction.cameras) == {3}
    np.testing.assert_allclose(
        reconstruction.camera_center(3),
        [-1.0, 0.0, 0.0],
    )

    landmark_id = reconstruction.add_landmark(
        xyz=np.array([0.0, 0.0, 5.0]),
        track_id=7,
        reprojection_error_px=0.25,
    )

    assert landmark_id == 0
    assert reconstruction.landmarks[0].track_id == 7
    np.testing.assert_allclose(
        reconstruction.points3d,
        [[0.0, 0.0, 5.0]],
    )


def test_apply_optimization_updates_typed_state():
    K = np.eye(3)
    reconstruction = Reconstruction(
        intrinsics=CameraIntrinsics.from_matrix(K)
    )
    reconstruction.add_camera(
        0,
        CameraPose(np.eye(3), np.zeros(3)),
    )
    reconstruction.add_landmark(
        np.array([0.0, 0.0, 2.0]),
        track_id=0,
    )

    new_pose = (
        np.eye(3),
        np.array([0.1, 0.2, 0.3]),
    )
    new_point = np.array([[1.0, 2.0, 3.0]])

    reconstruction.apply_optimization(
        {0: new_pose},
        new_point,
    )

    np.testing.assert_allclose(
        reconstruction.cameras[0].pose.t,
        [0.1, 0.2, 0.3],
    )
    np.testing.assert_allclose(
        reconstruction.landmarks[0].xyz,
        [1.0, 2.0, 3.0],
    )


def test_retain_landmarks_remaps_tracks():
    reconstruction = Reconstruction(
        intrinsics=CameraIntrinsics.from_matrix(np.eye(3))
    )

    tracks = [
        Track(
            id=0,
            observations={
                0: Observation(0, 1),
                1: Observation(1, 2),
            },
            point3d_id=0,
        ),
        Track(
            id=1,
            observations={
                0: Observation(0, 3),
                1: Observation(1, 4),
            },
            point3d_id=1,
        ),
    ]

    reconstruction.add_landmark(
        [0.0, 0.0, 3.0],
        track_id=0,
    )
    reconstruction.add_landmark(
        [1.0, 0.0, 4.0],
        track_id=1,
    )

    removed = reconstruction.retain_landmarks(
        np.array([False, True]),
        tracks,
    )

    assert removed == 1
    assert len(reconstruction.landmarks) == 1
    assert reconstruction.landmarks[0].id == 0
    assert reconstruction.landmarks[0].track_id == 1
    assert tracks[0].point3d_id is None
    assert tracks[1].point3d_id == 0


if __name__ == "__main__":
    test_reconstruction_camera_and_landmark_state()
    test_apply_optimization_updates_typed_state()
    test_retain_landmarks_remaps_tracks()
    print("test_models.py: ALL PASSED")
