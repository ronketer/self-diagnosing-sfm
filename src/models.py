"""Geometric and reconstruction data models.

WORLD -> CAMERA CONVENTION
--------------------------
Points are transformed from world coordinates to camera coordinates by:
    X_camera = R @ X_world + t

The camera center C in world coordinates satisfies:
    0 = R @ C + t  =>  C = -R.T @ t
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    K: np.ndarray
    distortion: np.ndarray | None = None

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

    @classmethod
    def from_matrix(cls, K: np.ndarray, distortion: np.ndarray | None = None) -> CameraIntrinsics:
        return cls(K=np.asarray(K, dtype=float), distortion=distortion)


@dataclass
class CameraPose:
    """Camera extrinsic pose representing transformation from World to Camera frame."""
    R: np.ndarray
    t: np.ndarray

    @property
    def center(self) -> np.ndarray:
        """Camera optical center in world coordinates: C = -R.T @ t."""
        return -self.R.T @ self.t

    def as_tuple(self) -> tuple[np.ndarray, np.ndarray]:
        return self.R, self.t


@dataclass
class RegisteredCamera:
    """A camera successfully registered into the reconstruction."""
    image_id: int
    pose: CameraPose


@dataclass
class Landmark:
    """A 3D scene point associated with a feature track."""
    id: int
    xyz: np.ndarray
    track_id: int


@dataclass
class Reconstruction:
    """Reconstruction state holding intrinsics, poses, and 3D points."""
    K: np.ndarray
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    points3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def camera_center(self, i: int) -> np.ndarray:
        R, t = self.poses[i]
        return -R.T @ t

    def get_pose(self, i: int) -> CameraPose:
        R, t = self.poses[i]
        return CameraPose(R=R, t=t)

    def set_pose(self, i: int, pose: CameraPose | tuple[np.ndarray, np.ndarray]) -> None:
        if isinstance(pose, CameraPose):
            self.poses[i] = (pose.R, pose.t)
        else:
            self.poses[i] = pose
