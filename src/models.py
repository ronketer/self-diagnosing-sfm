"""Canonical data models for the incremental Structure-from-Motion system.

Coordinate convention
---------------------
Camera poses always map world coordinates to camera coordinates:

    X_camera = R @ X_world + t

Therefore the camera center in world coordinates is:

    C = -R.T @ t

The classes in this module are the single source of truth for camera intrinsics,
camera poses, feature observations/tracks, landmarks, and reconstruction state.
Numerical solvers may use NumPy arrays internally, but conversions happen only
at solver boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SfMConfig
    from .tracks import TrackManager
    from .validation.diagnostics import ReconstructionDiagnostics


class UnsupportedCameraModelError(ValueError):
    """Raised when the active SfM camera model cannot represent the input."""


@dataclass(frozen=True)
class CameraIntrinsics:
    """Shared calibrated pinhole-camera intrinsics.

    The current MVP deliberately supports one camera model:
    - a shared 3x3 intrinsic matrix K for all views;
    - zero lens distortion, or images that were pre-undistorted.

    Keeping this assumption explicit is safer than silently ignoring a camera
    model that the reconstruction equations do not represent.
    """

    K: np.ndarray
    distortion: np.ndarray | None = None

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=float)
        if K.shape != (3, 3):
            raise ValueError(
                f"Camera intrinsic matrix K must have shape (3, 3), got {K.shape}."
            )
        if not np.all(np.isfinite(K)):
            raise ValueError("Camera intrinsic matrix K contains non-finite values.")

        distortion = (
            None
            if self.distortion is None
            else np.asarray(self.distortion, dtype=float).reshape(-1)
        )
        if distortion is not None and np.any(np.abs(distortion) > 1e-6):
            raise UnsupportedCameraModelError(
                f"Non-zero lens distortion detected: {distortion}. "
                "This SfM implementation requires ideal-pinhole or "
                "pre-undistorted input images."
            )

        object.__setattr__(self, "K", K)
        object.__setattr__(self, "distortion", distortion)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def validate_compatible(
        self,
        other: CameraIntrinsics | np.ndarray,
        tol: float = 1e-3,
    ) -> None:
        """Enforce the shared-intrinsics assumption."""

        other_intrinsics = (
            other
            if isinstance(other, CameraIntrinsics)
            else CameraIntrinsics.from_matrix(other)
        )

        if not np.allclose(self.K, other_intrinsics.K, atol=tol, rtol=0.0):
            raise UnsupportedCameraModelError(
                "Incompatible camera intrinsics detected.\n"
                f"Reference K:\n{self.K}\n"
                f"Other K:\n{other_intrinsics.K}\n"
                "This SfM implementation requires shared intrinsics across all views."
            )

    @classmethod
    def from_matrix(
        cls,
        K: np.ndarray,
        distortion: np.ndarray | None = None,
    ) -> CameraIntrinsics:
        """Construct and validate an intrinsic model."""

        return cls(K=np.asarray(K, dtype=float), distortion=distortion)


@dataclass
class CameraPose:
    """World-to-camera extrinsic pose: ``X_camera = R @ X_world + t``."""

    R: np.ndarray
    t: np.ndarray

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=float).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=float).reshape(3)

    @property
    def center(self) -> np.ndarray:
        """Camera optical center in world coordinates."""

        return -self.R.T @ self.t

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the NumPy representation expected by numerical solvers."""

        return self.R, self.t


@dataclass
class RegisteredCamera:
    """A camera that has been registered into the common reconstruction."""

    image_id: int
    pose: CameraPose


@dataclass(frozen=True)
class Observation:
    """One detected feature observation in one image.

    Pixel coordinates are not duplicated here. ``keypoint_id`` indexes the
    canonical keypoint array belonging to ``image_id``.
    """

    image_id: int
    keypoint_id: int


@dataclass
class Track:
    """A multi-view feature track.

    Invariant:
        at most one observation from each image.

    ``point3d_id`` is the ID of the Landmark created from this track, or None
    until the track has been successfully triangulated.
    """

    id: int
    observations: dict[int, Observation]
    point3d_id: int | None = None

    def __post_init__(self) -> None:
        for image_id, observation in self.observations.items():
            if image_id != observation.image_id:
                raise ValueError(
                    "Track observation key does not match Observation.image_id: "
                    f"key={image_id}, observation.image_id={observation.image_id}"
                )

    def has_image(self, image_id: int) -> bool:
        return image_id in self.observations

    def keypoint_id(self, image_id: int) -> int:
        return self.observations[image_id].keypoint_id

    @property
    def image_ids(self) -> set[int]:
        return set(self.observations)


@dataclass
class Landmark:
    """A reconstructed 3D scene point associated with one feature track."""

    id: int
    xyz: np.ndarray
    track_id: int
    reprojection_error_px: float | None = None

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(3)


@dataclass
class PairGeometry:
    """Verified two-view epipolar geometry for one image pair."""

    image_a: int
    image_b: int

    raw_matches: np.ndarray
    verified_matches: np.ndarray

    E: np.ndarray
    inlier_mask: np.ndarray

    n_raw_matches: int
    n_inliers: int
    inlier_ratio: float

    median_sampson_error: float | None = None
    score: float = 0.0


@dataclass
class Reconstruction:
    """Canonical geometric reconstruction.

    This object owns the authoritative camera and landmark state.  Convenience
    properties ``K``, ``poses`` and ``points3d`` expose NumPy views for
    numerical algorithms and existing analysis code, but they are derived from
    the typed domain objects rather than being independent sources of truth.
    """

    intrinsics: CameraIntrinsics
    cameras: dict[int, RegisteredCamera] = field(default_factory=dict)
    landmarks: list[Landmark] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        """Shared intrinsic matrix (read-only architectural adapter)."""

        return self.intrinsics.K

    @property
    def poses(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Solver-friendly world-to-camera pose dictionary."""

        return {
            image_id: camera.pose.as_tuple()
            for image_id, camera in self.cameras.items()
        }

    @property
    def points3d(self) -> np.ndarray:
        """Dense ``(N,3)`` landmark array ordered by landmark ID."""

        if not self.landmarks:
            return np.zeros((0, 3), dtype=float)
        return np.vstack([landmark.xyz for landmark in self.landmarks])

    def add_camera(self, image_id: int, pose: CameraPose) -> None:
        """Register or update a camera pose."""

        self.cameras[image_id] = RegisteredCamera(
            image_id=image_id,
            pose=pose,
        )

    def set_pose(
        self,
        image_id: int,
        pose: CameraPose | tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Compatibility helper used at solver boundaries."""

        if not isinstance(pose, CameraPose):
            pose = CameraPose(R=pose[0], t=pose[1])
        self.add_camera(image_id, pose)

    def get_pose(self, image_id: int) -> CameraPose:
        return self.cameras[image_id].pose

    def camera_center(self, image_id: int) -> np.ndarray:
        return self.cameras[image_id].pose.center

    def add_landmark(
        self,
        xyz: np.ndarray,
        track_id: int,
        reprojection_error_px: float | None = None,
    ) -> int:
        """Append one landmark and return its dense landmark ID."""

        landmark_id = len(self.landmarks)
        self.landmarks.append(
            Landmark(
                id=landmark_id,
                xyz=xyz,
                track_id=track_id,
                reprojection_error_px=reprojection_error_px,
            )
        )
        return landmark_id

    def apply_optimization(
        self,
        poses: dict[int, tuple[np.ndarray, np.ndarray] | CameraPose],
        points3d: np.ndarray,
    ) -> None:
        """Apply a numerical optimizer result back to canonical typed state."""

        if set(poses) != set(self.cameras):
            raise ValueError(
                "Optimizer camera IDs do not match reconstruction camera IDs: "
                f"{sorted(poses)} vs {sorted(self.cameras)}"
            )

        points = np.asarray(points3d, dtype=float).reshape(-1, 3)
        if len(points) != len(self.landmarks):
            raise ValueError(
                "Optimizer landmark count does not match reconstruction: "
                f"{len(points)} vs {len(self.landmarks)}"
            )

        for image_id, pose in poses.items():
            self.set_pose(image_id, pose)

        for landmark, xyz in zip(self.landmarks, points):
            landmark.xyz = np.asarray(xyz, dtype=float).reshape(3)

    def retain_landmarks(
        self,
        keep: np.ndarray,
        tracks: list[Track],
    ) -> int:
        """Remove rejected landmarks and remap every Track.point3d_id.

        Landmark IDs are intentionally dense because the bundle-adjustment
        observation format indexes points by integer ID.
        """

        keep = np.asarray(keep, dtype=bool).reshape(-1)
        if len(keep) != len(self.landmarks):
            raise ValueError(
                f"keep mask has {len(keep)} entries for "
                f"{len(self.landmarks)} landmarks"
            )

        remap = -np.ones(len(keep), dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()), dtype=np.int64)

        retained: list[Landmark] = []
        for old_id, landmark in enumerate(self.landmarks):
            if not keep[old_id]:
                continue
            new_id = int(remap[old_id])
            landmark.id = new_id
            retained.append(landmark)

        for track in tracks:
            if track.point3d_id is None:
                continue
            new_id = int(remap[track.point3d_id])
            track.point3d_id = new_id if new_id >= 0 else None

        removed = len(self.landmarks) - len(retained)
        self.landmarks = retained
        return removed


@dataclass
class ReconstructionState:
    """Mutable orchestration state for one incremental SfM run.

    Geometry itself is owned by ``reconstruction``.  This class stores the
    runtime context needed by the incremental algorithm: feature tracks,
    keypoints, configuration, diagnostics, registration order, and counters.
    """

    reconstruction: Reconstruction
    track_manager: "TrackManager"
    keypoints: list[np.ndarray]
    config: "SfMConfig"
    diagnostics: "ReconstructionDiagnostics"

    registration_order: list[int] = field(default_factory=list)
    anchor_camera: int | None = None
    scale_camera: int | None = None
    initial_baseline: float = 1.0

    pnp_attempt_count: int = 0
    pnp_accepted_count: int = 0
    pnp_rejected_count: int = 0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        return len(self.keypoints)

    @property
    def registered_images(self) -> set[int]:
        """Registered image IDs, derived from canonical camera state."""

        return set(self.reconstruction.cameras)

    def has_unregistered_images(self) -> bool:
        return len(self.reconstruction.cameras) < self.n_images

    def add_camera(self, image_id: int, pose: CameraPose) -> None:
        already_registered = image_id in self.reconstruction.cameras
        self.reconstruction.add_camera(image_id, pose)
        if not already_registered:
            self.registration_order.append(image_id)
