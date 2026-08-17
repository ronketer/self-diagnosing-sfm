"""Export a reconstruction to the standard COLMAP text model format."""

from pathlib import Path

import numpy as np


def R_to_qvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> COLMAP quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        w = np.sqrt(1 + t) / 2
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = np.empty(4)
        q[1 + i] = s / 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
        w, x, y, z = q
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def export_colmap_text(recon, tm, images, keypoints, point_rgb, out_dir: Path):
    """Write cameras.txt / images.txt / points3D.txt for registered views.

    point_rgb: (N,3) uint8 colors for recon.points3d.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    reg = sorted(recon.poses)
    K = recon.K
    h, w = images[reg[0]].gray.shape

    with open(out_dir / "cameras.txt", "w") as f:
        f.write("# Camera list: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")

    # per-image observation lists (x, y, point3d_id), and per-point tracks
    img_id = {img: k + 1 for k, img in enumerate(reg)}
    obs_per_image = {img: [] for img in reg}
    tracks_per_point: dict[int, list[tuple[int, int]]] = {}
    for tr in tm.tracks:
        if tr.point3d_id is None:
            continue
        for img, kp in tr.obs.items():
            if img not in img_id:
                continue
            idx = len(obs_per_image[img])
            x, y = keypoints[img][kp]
            obs_per_image[img].append((x, y, tr.point3d_id))
            tracks_per_point.setdefault(tr.point3d_id, []).append((img_id[img], idx))

    with open(out_dir / "images.txt", "w") as f:
        f.write("# Image list: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for img in reg:
            R, t = recon.poses[img]
            q = R_to_qvec(R)
            f.write(
                f"{img_id[img]} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} 1 {images[img].name}\n"
            )
            f.write(
                " ".join(f"{x:.3f} {y:.3f} {pid + 1}" for x, y, pid in obs_per_image[img])
                + "\n"
            )

    with open(out_dir / "points3D.txt", "w") as f:
        f.write("# 3D point list: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, X in enumerate(recon.points3d):
            track = tracks_per_point.get(pid, [])
            if not track:
                continue
            r, g, b = point_rgb[pid]
            f.write(
                f"{pid + 1} {X[0]:.9f} {X[1]:.9f} {X[2]:.9f} {r} {g} {b} 1.0 "
                + " ".join(f"{iid} {idx}" for iid, idx in track)
                + "\n"
            )
