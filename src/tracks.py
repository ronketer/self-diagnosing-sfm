"""Feature tracks: union-find over (image, keypoint) observations."""

import numpy as np


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        root = x
        while self.parent.setdefault(root, root) != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


class Track:
    __slots__ = ("obs", "point3d_id")

    def __init__(self, obs: dict[int, int]):
        self.obs = obs  # img_idx -> kp_idx
        self.point3d_id: int | None = None


class TrackManager:
    """Builds consistent multi-view tracks from a pairwise match graph."""

    def __init__(self, match_graph: dict[tuple[int, int], np.ndarray]):
        uf = UnionFind()
        for (i, j), matches in match_graph.items():
            for qi, ti in matches:
                uf.union((i, int(qi)), (j, int(ti)))

        groups: dict[tuple, list[tuple[int, int]]] = {}
        for obs in uf.parent:
            groups.setdefault(uf.find(obs), []).append(obs)

        self.tracks: list[Track] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            obs = {}
            consistent = True
            for img, kp in members:
                if img in obs:  # same image twice -> ambiguous track
                    consistent = False
                    break
                obs[img] = kp
            if consistent:
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
