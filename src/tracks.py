"""Feature tracks: conflict-aware union-find over (image, keypoint) observations."""

from __future__ import annotations

import numpy as np


class ConflictAwareUnionFind:
    """Union-Find data structure that enforces at most one observation per image per track."""

    def __init__(self):
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        # Maps root -> set of image indices present in the component
        self.images_per_root: dict[tuple[int, int], set[int]] = {}

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        if x not in self.parent:
            self.parent[x] = x
            self.images_per_root[x] = {x[0]}
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        curr = x
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        """Attempt to join components containing a and b.

        Returns True if merged, False if rejected due to conflict (same image).
        """
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return True  # Already connected

        # Conflict check: do both components contain observations from the same image?
        imgs_a = self.images_per_root[ra]
        imgs_b = self.images_per_root[rb]
        if imgs_a & imgs_b:
            return False  # Reject conflicting match edge

        # Merge rb into ra
        self.parent[rb] = ra
        self.images_per_root[ra] = imgs_a | imgs_b
        del self.images_per_root[rb]
        return True


class Track:
    __slots__ = ("obs", "point3d_id")

    def __init__(self, obs: dict[int, int]):
        self.obs = obs  # img_idx -> kp_idx
        self.point3d_id: int | None = None


class TrackManager:
    """Builds conflict-free multi-view tracks from a pairwise match graph."""

    def __init__(self, match_graph: dict[tuple[int, int], np.ndarray]):
        uf = ConflictAwareUnionFind()
        self.n_rejected_edges = 0

        for (i, j), matches in match_graph.items():
            for qi, ti in matches:
                merged = uf.union((i, int(qi)), (j, int(ti)))
                if not merged:
                    self.n_rejected_edges += 1

        groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for obs in uf.parent:
            groups.setdefault(uf.find(obs), []).append(obs)

        self.tracks: list[Track] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            obs = {img: kp for img, kp in members}
            self.tracks.append(Track(obs))

        # img_idx -> list of track indices observing it
        self._by_image: dict[int, list[int]] = {}
        for ti, tr in enumerate(self.tracks):
            for img in tr.obs:
                self._by_image.setdefault(img, []).append(ti)

    def tracks_seen_by(self, img_idx: int) -> list[int]:
        return self._by_image.get(img_idx, [])

    def correspondences_2d3d(
        self, img_idx: int, keypoints: list[np.ndarray], points3d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(pts2d, pts3d, track_ids) for tracks of img_idx that already have a
        3D point — the PnP input for registering a new image."""
        p2, p3, tids = [], [], []
        for ti in self.tracks_seen_by(img_idx):
            tr = self.tracks[ti]
            if tr.point3d_id is not None:
                p2.append(keypoints[img_idx][tr.obs[img_idx]])
                p3.append(points3d[tr.point3d_id])
                tids.append(ti)
        return (
            np.array(p2).reshape(-1, 2),
            np.array(p3).reshape(-1, 3),
            np.array(tids, dtype=np.int64),
        )

    def triangulatable_tracks(self, registered: set[int]) -> list[int]:
        """Untriangulated tracks visible in >= 2 registered images."""
        out = []
        for ti, tr in enumerate(self.tracks):
            if tr.point3d_id is None and len(registered & tr.obs.keys()) >= 2:
                out.append(ti)
        return out
