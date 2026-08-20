"""SIFT feature detection and pairwise descriptor matching."""

from pathlib import Path

import cv2
import numpy as np


def detect_features(gray: np.ndarray, nfeatures: int = 8000):
    """Return (keypoints (N,2) float32, descriptors (N,128))."""
    sift = cv2.SIFT_create(
        nfeatures=nfeatures, contrastThreshold=0.01, edgeThreshold=20
    )
    kps, desc = sift.detectAndCompute(gray, None)
    pts = np.array([kp.pt for kp in kps], dtype=np.float64)
    return pts, desc


def match_pair(desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """Lowe-ratio + mutual-consistency matching. Returns (M,2) index pairs."""
    bf = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_matches(d1, d2):
        out = {}
        for m, n in bf.knnMatch(d1, d2, k=2):
            if m.distance < ratio * n.distance:
                out[m.queryIdx] = m.trainIdx
        return out

    fwd = ratio_matches(desc1, desc2)
    bwd = ratio_matches(desc2, desc1)
    pairs = [(q, t) for q, t in fwd.items() if bwd.get(t) == q]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def build_match_graph(
    descriptors: list[np.ndarray],
    min_matches: int = 30,
    cache_path: Path | None = None,
    max_gap: int | None = None,
) -> dict[tuple[int, int], np.ndarray]:
    """Match image pairs (i < j), optionally only within a sequential window
    |i - j| <= max_gap (for ordered ring/sequence captures, this avoids false
    matches between symmetric structures on far sides of the object).
    Cached to an .npz keyed 'i_j'."""
    if cache_path is not None and cache_path.exists():
        data = np.load(cache_path)
        return {
            tuple(map(int, k.split("_"))): data[k] for k in data.files
        }

    n = len(descriptors)
    graph = {}
    for i in range(n):
        for j in range(i + 1, n):
            if max_gap is not None and j - i > max_gap:
                continue
            m = match_pair(descriptors[i], descriptors[j])
            if len(m) >= min_matches:
                graph[(i, j)] = m

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **{f"{i}_{j}": m for (i, j), m in graph.items()})
    return graph
