"""Feature tracks built from pairwise feature correspondences.

Tracks are constructed using a conflict-aware Union-Find.

TRACK INVARIANT
---------------
A single physical feature track may contain at most one observation
from each image.

When conflicting match edges exist, higher-confidence geometric edges
are processed first so that a weak false match is less likely to block
a stronger consistent match.
"""

from __future__ import annotations

import numpy as np

from .geometry.essential import sampson_distance
from .geometry.projection import to_normalized
from .models import Observation, PairGeometry, Track


class ConflictAwareUnionFind:
    """Union-Find enforcing at most one observation per image/component."""

    def __init__(self) -> None:
        self.parent: dict[
            tuple[int, int],
            tuple[int, int],
        ] = {}

        # root -> image IDs represented by the component
        self.images_per_root: dict[
            tuple[int, int],
            set[int],
        ] = {}

    def find(
        self,
        x: tuple[int, int],
    ) -> tuple[int, int]:
        """Return the component root for an observation.

        Observations are represented as:

            (image_id, keypoint_id)
        """

        if x not in self.parent:
            self.parent[x] = x
            self.images_per_root[x] = {x[0]}
            return x

        root = x

        while self.parent[root] != root:
            root = self.parent[root]

        # Path compression.
        current = x

        while current != root:
            nxt = self.parent[current]
            self.parent[current] = root
            current = nxt

        return root

    def union(
        self,
        a: tuple[int, int],
        b: tuple[int, int],
    ) -> bool:
        """Attempt to merge the components containing ``a`` and ``b``.

        Returns
        -------
        True
            The edge is compatible with the track invariant.

        False
            Merging the components would create two observations
            from the same image.
        """

        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            # Already connected.
            return True

        images_a = self.images_per_root[root_a]
        images_b = self.images_per_root[root_b]

        # If both components already contain any common image, the
        # merge would produce two observations from that image.
        if images_a & images_b:
            return False

        # Merge root_b into root_a.
        self.parent[root_b] = root_a

        self.images_per_root[root_a] = (
            images_a | images_b
        )

        del self.images_per_root[root_b]

        return True


class TrackManager:
    """Build conflict-free multi-view feature tracks.

    Parameters
    ----------
    match_graph:
        Mapping from image pair to either:

            np.ndarray([[kp_i, kp_j], ...])

        or:

            PairGeometry

        PairGeometry is preferred in production because it contains
        geometrically verified matches and the corresponding Essential
        Matrix.

    keypoints:
        Per-image keypoint coordinates.

        Required when PairGeometry is supplied and Sampson-error-based
        confidence ordering should be calculated.

    K:
        Shared camera intrinsic matrix.

        Required with ``keypoints`` for Sampson-error confidence.

    edge_confidences:
        Optional explicit confidence arrays for raw match arrays.

        Mapping:

            (image_i, image_j) -> confidence_per_match

        Higher values mean more reliable edges.
    """

    def __init__(
        self,
        match_graph: dict[
            tuple[int, int],
            np.ndarray | PairGeometry,
        ],
        keypoints: list[np.ndarray] | None = None,
        K: np.ndarray | None = None,
        edge_confidences: dict[
            tuple[int, int],
            np.ndarray,
        ]
        | None = None,
    ) -> None:
        union_find = ConflictAwareUnionFind()

        self.n_rejected_edges = 0

        # Each candidate is:
        #
        # (
        #     confidence,
        #     (image_i, keypoint_i),
        #     (image_j, keypoint_j),
        # )
        #
        # All edges are collected before merging so ordering is global
        # across all image pairs.
        candidate_edges: list[
            tuple[
                float,
                tuple[int, int],
                tuple[int, int],
            ]
        ] = []

        for pair_key, value in match_graph.items():
            image_i, image_j = pair_key

            if isinstance(value, PairGeometry):
                matches = np.asarray(
                    value.verified_matches
                ).reshape(-1, 2)

                # Preferred production confidence:
                #
                # smaller Sampson error -> larger confidence
                if (
                    keypoints is not None
                    and K is not None
                    and len(matches) > 0
                ):
                    points_i = keypoints[image_i][
                        matches[:, 0].astype(int)
                    ]

                    points_j = keypoints[image_j][
                        matches[:, 1].astype(int)
                    ]

                    normalized_i = to_normalized(
                        points_i,
                        K,
                    )

                    normalized_j = to_normalized(
                        points_j,
                        K,
                    )

                    sampson_errors = sampson_distance(
                        value.E,
                        normalized_i,
                        normalized_j,
                    )

                    scores = 1.0 / (
                        1.0
                        + np.asarray(
                            sampson_errors,
                            dtype=float,
                        )
                    )

                else:
                    # PairGeometry has already passed geometric
                    # verification, but no per-edge metric can be
                    # computed without coordinates + calibration.
                    scores = np.ones(
                        len(matches),
                        dtype=float,
                    )

            else:
                matches = np.asarray(
                    value
                ).reshape(-1, 2)

                if (
                    edge_confidences is not None
                    and pair_key in edge_confidences
                ):
                    scores = np.asarray(
                        edge_confidences[pair_key],
                        dtype=float,
                    ).reshape(-1)

                    if len(scores) != len(matches):
                        raise ValueError(
                            "Confidence count mismatch for pair "
                            f"{pair_key}: "
                            f"{len(scores)} confidence values for "
                            f"{len(matches)} matches."
                        )

                else:
                    # No explicit confidence information.
                    scores = np.ones(
                        len(matches),
                        dtype=float,
                    )

            if len(scores) != len(matches):
                raise ValueError(
                    "Internal confidence/match count mismatch for "
                    f"pair {pair_key}: "
                    f"{len(scores)} confidence values for "
                    f"{len(matches)} matches."
                )

            for match_idx in range(len(matches)):
                query_idx, train_idx = matches[match_idx]

                confidence = float(
                    scores[match_idx]
                )

                candidate_edges.append(
                    (
                        confidence,
                        (
                            int(image_i),
                            int(query_idx),
                        ),
                        (
                            int(image_j),
                            int(train_idx),
                        ),
                    )
                )

        # Highest confidence first.
        #
        # The additional image/keypoint fields provide deterministic
        # tie-breaking when two edges have exactly the same score.
        candidate_edges.sort(
            key=lambda edge: (
                -edge[0],
                edge[1][0],
                edge[1][1],
                edge[2][0],
                edge[2][1],
            )
        )

        for _confidence, observation_a, observation_b in candidate_edges:
            merged = union_find.union(
                observation_a,
                observation_b,
            )

            if not merged:
                self.n_rejected_edges += 1

        # Build connected components.
        groups: dict[
            tuple[int, int],
            list[tuple[int, int]],
        ] = {}

        for observation in union_find.parent:
            root = union_find.find(
                observation
            )

            groups.setdefault(
                root,
                [],
            ).append(observation)

        self.tracks: list[Track] = []

        for members in groups.values():
            # A valid multi-view track requires at least two
            # observations.
            if len(members) < 2:
                continue

            observations = {
                image_id: Observation(
                    image_id=image_id,
                    keypoint_id=keypoint_id,
                )
                for image_id, keypoint_id in members
            }

            # Defensive invariant check.
            #
            # ConflictAwareUnionFind should already guarantee this.
            if len(observations) != len(members):
                raise RuntimeError(
                    "Track construction produced multiple observations "
                    "from the same image."
                )

            track_id = len(
                self.tracks
            )

            self.tracks.append(
                Track(
                    id=track_id,
                    observations=observations,
                )
            )

        # image_id -> track IDs observed in that image
        self._by_image: dict[
            int,
            list[int],
        ] = {}

        for track_id, track in enumerate(
            self.tracks
        ):
            for image_id in track.image_ids:
                self._by_image.setdefault(
                    image_id,
                    [],
                ).append(track_id)

    def tracks_seen_by(
        self,
        image_id: int,
    ) -> list[int]:
        """Return IDs of tracks containing an observation in an image."""

        return self._by_image.get(
            image_id,
            [],
        )

    def correspondences_2d3d(
        self,
        image_id: int,
        keypoints: list[np.ndarray],
        points3d: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Collect 2D↔3D correspondences for camera registration.

        A track contributes if:
        - it is visible in ``image_id``;
        - it already has a reconstructed 3D point.

        Returns
        -------
        pts2d:
            Shape ``(N, 2)``.

        pts3d:
            Shape ``(N, 3)``.

        track_ids:
            Shape ``(N,)``.
        """

        points_2d: list[np.ndarray] = []
        points_3d: list[np.ndarray] = []
        track_ids: list[int] = []

        for track_id in self.tracks_seen_by(
            image_id
        ):
            track = self.tracks[
                track_id
            ]

            if track.point3d_id is None:
                continue

            keypoint_id = track.keypoint_id(
                image_id
            )

            points_2d.append(
                keypoints[image_id][
                    keypoint_id
                ]
            )

            points_3d.append(
                points3d[
                    track.point3d_id
                ]
            )

            track_ids.append(
                track_id
            )

        return (
            np.asarray(
                points_2d,
                dtype=float,
            ).reshape(-1, 2),
            np.asarray(
                points_3d,
                dtype=float,
            ).reshape(-1, 3),
            np.asarray(
                track_ids,
                dtype=np.int64,
            ),
        )

    def triangulatable_tracks(
        self,
        registered: set[int],
    ) -> list[int]:
        """Return untriangulated tracks seen by >=2 registered cameras."""

        result: list[int] = []

        for track_id, track in enumerate(
            self.tracks
        ):
            if track.point3d_id is not None:
                continue

            registered_observations = (
                registered
                & track.image_ids
            )

            if len(
                registered_observations
            ) >= 2:
                result.append(
                    track_id
                )

        return result