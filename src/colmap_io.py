"""Minimal COLMAP binary model reader (cameras.bin / images.bin).

Format reference: COLMAP src/colmap/scene/reconstruction_io.cc
"""

import struct
from pathlib import Path
import numpy as np

from .models import UnsupportedCameraModelError

# camera model id -> (name, n_params)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),  # f, cx, cy
    1: ("PINHOLE", 4),  # fx, fy, cx, cy
    2: ("SIMPLE_RADIAL", 4),  # f, cx, cy, k
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
}


def read_cameras_bin(path: Path) -> dict[int, dict]:
    cameras = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            cam_id, model_id, w, h = struct.unpack("<iiQQ", f.read(24))
            name, n_params = CAMERA_MODELS[model_id]
            params = struct.unpack(f"<{n_params}d", f.read(8 * n_params))
            cameras[cam_id] = {
                "model": name,
                "width": w,
                "height": h,
                "params": np.array(params),
            }
    return cameras


def qvec_to_R(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def read_images_bin(path: Path) -> dict[str, dict]:
    """Returns name -> {R, t, camera_id} with x = K[R|t]X convention."""
    images = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<4d", f.read(32)))
            tvec = np.array(struct.unpack("<3d", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n_pts = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * n_pts)  # skip 2D points (x, y, point3D_id)
            images[name.decode()] = {
                "R": qvec_to_R(qvec),
                "t": tvec,
                "camera_id": camera_id,
            }
    return images


def camera_K(cam: dict) -> np.ndarray:
    p = cam["params"]
    if cam["model"] == "SIMPLE_PINHOLE":
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
    if cam["model"] == "PINHOLE":
        return np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    if cam["model"] in ("SIMPLE_RADIAL", "RADIAL", "OPENCV"):
        # If radial distortion parameters are non-zero, check tolerance
        distortion_params = p[3:] if cam["model"] == "PINHOLE" else p[4:] if cam["model"] == "OPENCV" else p[3:]
        if np.any(np.abs(distortion_params) > 1e-4):
            raise UnsupportedCameraModelError(
                f"COLMAP camera {cam['model']} specifies non-zero distortion parameters {distortion_params}. "
                f"This SfM implementation requires pre-undistorted images and zero distortion."
            )
        return np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]])
    raise UnsupportedCameraModelError(f"Unsupported COLMAP camera model {cam['model']}")
