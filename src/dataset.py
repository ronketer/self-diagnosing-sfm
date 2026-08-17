"""Dataset loading for the Middlebury multi-view stereo temple datasets."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import CameraIntrinsics, UnsupportedCameraModelError


@dataclass
class CameraGT:
    """Ground-truth camera: x = K [R | t] X (world-to-camera)."""

    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class Image:
    name: str
    gray: np.ndarray
    color: np.ndarray  # RGB


def parse_par_file(path: Path) -> dict[str, CameraGT]:
    cameras = {}
    lines = path.read_text().strip().splitlines()
    n = int(lines[0])
    K_ref = None
    for line in lines[1 : n + 1]:
        parts = line.split()
        name = parts[0]
        vals = np.array([float(v) for v in parts[1:]])
        K = vals[0:9].reshape(3, 3)
        R = vals[9:18].reshape(3, 3)
        t = vals[18:21]

        if K_ref is None:
            K_ref = K
        else:
            if not np.allclose(K_ref, K, atol=1e-3):
                raise UnsupportedCameraModelError(
                    f"Incompatible calibration in {path.name} for {name}: intrinsics differ from reference view."
                )

        cameras[name] = CameraGT(K=K, R=R, t=t)
    return cameras


def load_phone_folder(
    data_dir: Path, max_dim: int = 1024
) -> tuple[list[Image], None, CameraIntrinsics]:
    """Load a folder of phone photos (no ground truth). Intrinsics come from
    the EXIF 35mm-equivalent focal length: f_px = long_side * f35 / 36.
    Images are downscaled so their long side is max_dim."""
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS

    paths = sorted(
        p for p in data_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not paths:
        raise FileNotFoundError(f"no images in {data_dir}")

    f35_file = data_dir / "f35_equiv.txt"
    if f35_file.exists():
        f35 = float(f35_file.read_text().strip())
    else:
        exif = PILImage.open(paths[0])._getexif() or {}
        f35 = next(
            (v for k, v in exif.items() if TAGS.get(k) == "FocalLengthIn35mmFilm"), None
        )
        if f35 is None:
            raise ValueError("no EXIF 35mm focal length; calibrate the camera instead")

    images = []
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise FileNotFoundError(p)
        h, w = bgr.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        images.append(
            Image(
                name=p.name,
                gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                color=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )
        )

    h, w = images[0].gray.shape
    f_px = max(h, w) * float(f35) / 36.0
    K = np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1]])
    return images, None, CameraIntrinsics.from_matrix(K)


def load_colmap_scene(
    data_dir: Path, step: int = 3
) -> tuple[list[Image], list[CameraGT], CameraIntrinsics]:
    """Load a COLMAP-calibrated scene (images/ + sparse/0/*.bin), taking every
    `step`-th image. COLMAP poses serve as the reference "ground truth"."""
    from .colmap_io import camera_K, read_cameras_bin, read_images_bin

    sparse = data_dir / "sparse" / "0"
    if not sparse.exists():
        sparse = data_dir / "sparse"
    cams = read_cameras_bin(sparse / "cameras.bin")
    poses = read_images_bin(sparse / "images.bin")

    names = sorted(poses)[::step]
    images, gts = [], []
    K_ref = None
    for name in names:
        bgr = cv2.imread(str(data_dir / "images" / name))
        if bgr is None:
            raise FileNotFoundError(data_dir / "images" / name)
        images.append(
            Image(
                name=name,
                gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                color=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )
        )
        info = poses[name]
        cam = cams[info["camera_id"]]
        K_cur = camera_K(cam)
        # distributed images may be downsampled relative to the calibration
        h, w = images[-1].gray.shape
        K_cur = K_cur.copy()
        K_cur[0] *= w / cam["width"]
        K_cur[1] *= h / cam["height"]

        if K_ref is None:
            K_ref = K_cur
        else:
            if not np.allclose(K_ref, K_cur, atol=1e-2):
                raise UnsupportedCameraModelError(
                    f"COLMAP images specify incompatible camera intrinsics across views:\n{K_ref}\nvs\n{K_cur}"
                )

        gts.append(CameraGT(K=K_cur, R=info["R"], t=info["t"]))

    return images, gts, CameraIntrinsics.from_matrix(K_ref)


def load_dataset(data_dir: Path) -> tuple[list[Image], list[CameraGT] | None, CameraIntrinsics]:
    """Load images (sorted by name) and ground-truth cameras.

    Returns ``(images, gt_cameras, intrinsics)`` with one shared camera model.
    """
    if (data_dir / "sparse").exists():
        return load_colmap_scene(data_dir)
    par_files = list(data_dir.glob("*_par.txt"))
    if not par_files:
        # no Middlebury calibration file -> treat as a phone capture
        return load_phone_folder(data_dir)
    cameras = parse_par_file(par_files[0])

    images, gts = [], []
    for name in sorted(cameras):
        img_path = data_dir / name
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(img_path)
        images.append(
            Image(
                name=name,
                gray=cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY),
                color=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            )
        )
        gts.append(cameras[name])

    intrinsics = CameraIntrinsics.from_matrix(gts[0].K.copy())
    return images, gts, intrinsics
