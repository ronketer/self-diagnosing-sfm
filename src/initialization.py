"""Two-view initialization selection with explicit geometric evaluation.

Initialization has two responsibilities:

1. Find a geometrically reliable two-view reconstruction.
2. Prefer a pair that can seed the subsequent incremental SfM process.

The second criterion is important because the geometrically strongest pair
is not necessarily connected to enough additional observations to register
a third camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import SfMConfig
from .geometry.essential import (
    choose_pose,
    decompose_essential,
    ransac_essential,
)
from .geometry.projection import (
    reproject,
    to_normalized,
    triangulation_angles_deg,
)
from .models import PairGeometry, Track
from .triangulation import triangulate_two_view
from .validation.degeneracy import evaluate_homography_support
from .validation.diagnostics import FailureReason


@dataclass
class InitialPairEvaluation:
    """Detailed geometric evaluation of one initialization-pair candidate."""

    image_a: int
    image_b: int

    n_matches: int
    n_inliers: int
    inlier_ratio: float

    cheirality_ratio: float
    median_parallax_deg: float
    median_reprojection_px: float

    homography_inliers: int = 0
    homography_inlier_ratio: float = 0.0
    homography_dominance_ratio: float = 0.0
    is_homography_dominant: bool = False

    # Incremental-SfM expansion information.
    #
    # These fields are deliberately separate from the geometric score.
    # Geometry determines whether a pair is valid.
    # Expansion support helps decide WHICH valid pair should initialize
    # the incremental reconstruction.
    expandable_camera_count: int = 0
    best_next_image: int | None = None
    best_next_correspondences: int = 0

    accepted: bool = False
    rejection_reason: FailureReason | str | None = None

    # Geometric score only.
    #
    # At present this is median parallax. It is intentionally not polluted
    # with graph-connectivity heuristics so diagnostics remain interpretable.
    score: float = 0.0

    E: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the evaluation for diagnostics.json."""

        return {
            "pair": [
                int(self.image_a),
                int(self.image_b),
            ],
            "n_matches": int(self.n_matches),
            "n_inliers": int(self.n_inliers),
            "inlier_ratio": round(float(self.inlier_ratio), 4),
            "cheirality_ratio": round(float(self.cheirality_ratio), 4),
            "median_parallax_deg": round(
                float(self.median_parallax_deg),
                3,
            ),
            "median_reprojection_px": round(
                float(self.median_reprojection_px),
                3,
            ),
            "homography_inliers": int(self.homography_inliers),
            "homography_inlier_ratio": round(
                float(self.homography_inlier_ratio),
                4,
            ),
            "homography_dominance_ratio": round(
                float(self.homography_dominance_ratio),
                4,
            ),
            "is_homography_dominant": bool(
                self.is_homography_dominant
            ),
            "expandable_camera_count": int(
                self.expandable_camera_count
            ),
            "best_next_image": (
                int(self.best_next_image)
                if self.best_next_image is not None
                else None
            ),
            "best_next_correspondences": int(
                self.best_next_correspondences
            ),
            "accepted": bool(self.accepted),
            "rejection_reason": (
                self.rejection_reason.value
                if isinstance(
                    self.rejection_reason,
                    FailureReason,
                )
                else self.rejection_reason
            ),
            "score": round(float(self.score), 4),
        }


def _sampson_thresh(
    K: np.ndarray,
    px: float = 3.0,
) -> float:
    """Convert an approximate pixel threshold to normalized coordinates."""

    f = (
        float(K[0, 0])
        + float(K[1, 1])
    ) / 2.0

    return (px / f) ** 2


def evaluate_pair(
    i: int,
    j: int,
    matches: np.ndarray,
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig | None = None,
) -> InitialPairEvaluation:
    """Evaluate a raw pair.

    This is the fallback / ablation API.

    In the normal production pipeline, Essential geometry has already been
    estimated during geometric verification and ``evaluate_pair_geometry()``
    is used instead so Essential RANSAC is not repeated.
    """

    if config is None:
        config = SfMConfig()

    matches = np.asarray(
        matches,
        dtype=np.int64,
    ).reshape(-1, 2)

    n_matches = len(matches)

    if n_matches < config.min_verified_matches:
        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=0,
            inlier_ratio=0.0,
            cheirality_ratio=0.0,
            median_parallax_deg=0.0,
            median_reprojection_px=0.0,
            accepted=False,
            rejection_reason=FailureReason.INSUFFICIENT_MATCHES,
        )

    x1 = to_normalized(
        keypoints[i][matches[:, 0]],
        K,
    )

    x2 = to_normalized(
        keypoints[j][matches[:, 1]],
        K,
    )

    thresh = _sampson_thresh(
        K,
        px=config.essential_ransac_px,
    )

    # Deterministic pair-specific seed.
    pair_seed = int(
        (
            i * 10007
            + j * 997
            + config.global_seed
        )
        % (2**31 - 1)
    )

    try:
        E, inliers = ransac_essential(
            x1,
            x2,
            thresh=thresh,
            iters=config.essential_ransac_iters,
            seed=pair_seed,
        )

    except RuntimeError:
        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=0,
            inlier_ratio=0.0,
            cheirality_ratio=0.0,
            median_parallax_deg=0.0,
            median_reprojection_px=0.0,
            accepted=False,
            rejection_reason=(
                FailureReason.ESSENTIAL_ESTIMATION_FAILED
            ),
        )

    inliers = np.asarray(
        inliers,
        dtype=bool,
    )

    n_inliers = int(
        inliers.sum()
    )

    pair_geometry = PairGeometry(
        image_a=i,
        image_b=j,
        raw_matches=matches,
        verified_matches=matches[inliers],
        E=E,
        inlier_mask=inliers,
        n_raw_matches=n_matches,
        n_inliers=n_inliers,
        inlier_ratio=float(
            n_inliers / max(n_matches, 1)
        ),
    )

    return evaluate_pair_geometry(
        pair_geometry,
        keypoints,
        K,
        config,
    )


def evaluate_pair_geometry(
    pg: PairGeometry,
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig,
) -> InitialPairEvaluation:
    """Evaluate already-estimated two-view geometry.

    Essential RANSAC is NOT repeated here.
    """

    i = pg.image_a
    j = pg.image_b

    n_matches = pg.n_raw_matches
    n_inliers = pg.n_inliers
    inlier_ratio = pg.inlier_ratio

    if (
        n_matches < config.min_verified_matches
        or n_inliers < config.min_init_inliers
    ):
        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            cheirality_ratio=0.0,
            median_parallax_deg=0.0,
            median_reprojection_px=0.0,
            accepted=False,
            rejection_reason=FailureReason.INSUFFICIENT_MATCHES,
            score=0.0,
            E=pg.E,
            inlier_mask=pg.inlier_mask,
        )

    verified_matches = np.asarray(
        pg.verified_matches,
        dtype=np.int64,
    ).reshape(-1, 2)

    pts_px1 = keypoints[i][
        verified_matches[:, 0]
    ]

    pts_px2 = keypoints[j][
        verified_matches[:, 1]
    ]

    # ------------------------------------------------------------------
    # 1. Homography-vs-Essential model evidence
    # ------------------------------------------------------------------

    (
        _H,
        _h_mask,
        h_inliers,
        h_ratio,
    ) = evaluate_homography_support(
        pts_px1,
        pts_px2,
        thresh_px=config.homography_ransac_px,
    )

    h_dominance = float(
        h_inliers
        / max(n_inliers, 1)
    )

    # ------------------------------------------------------------------
    # 2. Essential decomposition + cheirality
    # ------------------------------------------------------------------

    x1 = to_normalized(
        pts_px1,
        K,
    )

    x2 = to_normalized(
        pts_px2,
        K,
    )

    candidates = decompose_essential(
        pg.E
    )

    R, t, cheirality_mask = choose_pose(
        candidates,
        x1,
        x2,
    )

    cheirality_mask = np.asarray(
        cheirality_mask,
        dtype=bool,
    )

    cheirality_ratio = float(
        cheirality_mask.sum()
        / max(len(x1), 1)
    )

    if (
        cheirality_ratio
        < config.min_cheirality_ratio
    ):
        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            cheirality_ratio=cheirality_ratio,
            median_parallax_deg=0.0,
            median_reprojection_px=0.0,
            homography_inliers=h_inliers,
            homography_inlier_ratio=h_ratio,
            homography_dominance_ratio=h_dominance,
            is_homography_dominant=(
                h_dominance
                >= config.homography_dominance_ratio
            ),
            accepted=False,
            rejection_reason=FailureReason.POOR_CHEIRALITY,
            score=0.0,
            E=pg.E,
            R=R,
            t=t,
            inlier_mask=pg.inlier_mask,
        )

    # ------------------------------------------------------------------
    # 3. Initial two-view triangulation
    # ------------------------------------------------------------------

    P1 = np.hstack(
        [
            np.eye(3),
            np.zeros((3, 1)),
        ]
    )

    P2 = np.hstack(
        [
            R,
            t.reshape(3, 1),
        ]
    )

    X = triangulate_two_view(
        P1,
        P2,
        x1[cheirality_mask],
        x2[cheirality_mask],
    )

    # ------------------------------------------------------------------
    # 4. Parallax
    # ------------------------------------------------------------------

    C1 = np.zeros(
        3,
        dtype=float,
    )

    C2 = (
        -R.T
        @ t
    )

    angles = triangulation_angles_deg(
        C1,
        C2,
        X,
    )

    median_parallax = float(
        np.median(angles)
    )

    is_homography_dominant = bool(
        h_dominance
        >= config.homography_dominance_ratio
        and median_parallax
        < config.min_parallax_deg
    )

    if (
        median_parallax
        < config.min_parallax_deg
    ):
        reason = (
            FailureReason.HOMOGRAPHY_DOMINANT
            if is_homography_dominant
            else FailureReason.LOW_PARALLAX
        )

        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            cheirality_ratio=cheirality_ratio,
            median_parallax_deg=median_parallax,
            median_reprojection_px=0.0,
            homography_inliers=h_inliers,
            homography_inlier_ratio=h_ratio,
            homography_dominance_ratio=h_dominance,
            is_homography_dominant=is_homography_dominant,
            accepted=False,
            rejection_reason=reason,
            score=0.0,
            E=pg.E,
            R=R,
            t=t,
            inlier_mask=pg.inlier_mask,
        )

    # ------------------------------------------------------------------
    # 5. Pixel reprojection quality
    # ------------------------------------------------------------------

    proj1 = reproject(
        K,
        np.eye(3),
        np.zeros(3),
        X,
    )

    proj2 = reproject(
        K,
        R,
        t,
        X,
    )

    errs1 = np.linalg.norm(
        proj1
        - pts_px1[cheirality_mask],
        axis=1,
    )

    errs2 = np.linalg.norm(
        proj2
        - pts_px2[cheirality_mask],
        axis=1,
    )

    median_reproj = float(
        np.median(
            np.maximum(
                errs1,
                errs2,
            )
        )
    )

    if (
        median_reproj
        > config.max_triangulation_error_px
    ):
        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            cheirality_ratio=cheirality_ratio,
            median_parallax_deg=median_parallax,
            median_reprojection_px=median_reproj,
            homography_inliers=h_inliers,
            homography_inlier_ratio=h_ratio,
            homography_dominance_ratio=h_dominance,
            is_homography_dominant=is_homography_dominant,
            accepted=False,
            rejection_reason=(
                FailureReason.HIGH_REPROJECTION_ERROR
            ),
            score=0.0,
            E=pg.E,
            R=R,
            t=t,
            inlier_mask=pg.inlier_mask,
        )

    # Keep this score purely geometric.
    score = median_parallax

    return InitialPairEvaluation(
        image_a=i,
        image_b=j,
        n_matches=n_matches,
        n_inliers=n_inliers,
        inlier_ratio=inlier_ratio,
        cheirality_ratio=cheirality_ratio,
        median_parallax_deg=median_parallax,
        median_reprojection_px=median_reproj,
        homography_inliers=h_inliers,
        homography_inlier_ratio=h_ratio,
        homography_dominance_ratio=h_dominance,
        is_homography_dominant=is_homography_dominant,
        accepted=True,
        rejection_reason=None,
        score=score,
        E=pg.E,
        R=R,
        t=t,
        inlier_mask=pg.inlier_mask,
    )


def initial_pair_expansion_support(
    image_a: int,
    image_b: int,
    tracks: list[Track],
    min_correspondences: int,
) -> tuple[int, int | None, int]:
    """Estimate whether an initialization pair can grow incrementally.

    A future camera potentially gets a 2D<->3D correspondence from a
    feature track when that track is observed in:

        initialization image A
        initialization image B
        future image C

    Tracks observed in A+B can become triangulated landmarks after
    initialization. If the same tracks are also visible in C, they can
    support PnP registration of C.

    This is intentionally an estimate made BEFORE triangulation. Some
    tracks may later fail triangulation validation.

    Returns
    -------
    expandable_camera_count:
        Number of other cameras having at least ``min_correspondences``
        potential correspondences.

    best_next_image:
        Camera with the strongest potential support.

    best_next_correspondences:
        Number of potential 2D<->3D correspondences for that camera.
    """

    support: dict[int, int] = {}

    pair_images = {
        image_a,
        image_b,
    }

    for track in tracks:
        if not (
            track.has_image(image_a)
            and track.has_image(image_b)
        ):
            continue

        other_images = (
            track.image_ids
            - pair_images
        )

        for image_id in other_images:
            support[image_id] = (
                support.get(
                    image_id,
                    0,
                )
                + 1
            )

    if not support:
        return (
            0,
            None,
            0,
        )

    # Strongest immediate next-camera candidate.
    #
    # ``-image_id`` gives deterministic tie-breaking in favor of
    # the lower image ID.
    best_next_image, best_support = max(
        support.items(),
        key=lambda item: (
            item[1],
            -item[0],
        ),
    )

    expandable_camera_count = sum(
        count >= min_correspondences
        for count in support.values()
    )

    return (
        int(expandable_camera_count),
        int(best_next_image),
        int(best_support),
    )


def _initial_pair_selection_key(
    evaluation: InitialPairEvaluation,
) -> tuple[
    int,
    int,
    int,
    float,
    int,
    float,
    int,
    int,
]:
    """Ranking policy among ALREADY GEOMETRICALLY VALID pairs.

    Priority:
        1. Can this pair expand at all?
        2. How many cameras have enough potential PnP support?
        3. How strong is the best immediate next-camera support?
        4. Geometric score / parallax.
        5. Essential inlier count.
        6. Essential inlier ratio.
        7. Deterministic image-ID tie breaking.

    Geometry still determines acceptance/rejection before this ranking.
    """

    has_expandable_camera = int(
        evaluation.expandable_camera_count > 0
    )

    return (
        has_expandable_camera,
        evaluation.expandable_camera_count,
        evaluation.best_next_correspondences,
        float(evaluation.score),
        int(evaluation.n_inliers),
        float(evaluation.inlier_ratio),
        -int(evaluation.image_a),
        -int(evaluation.image_b),
    )


def choose_best_initial_pair_evaluation(
    evaluations: list[InitialPairEvaluation],
    tracks: list[Track] | None,
    min_pnp_correspondences: int,
) -> InitialPairEvaluation:
    """Choose the best accepted initialization evaluation.

    If tracks are available, expansion support is used as the primary
    ranking signal among geometrically valid pairs.

    If tracks are not supplied, this falls back to geometric ranking.
    """

    accepted = [
        evaluation
        for evaluation in evaluations
        if evaluation.accepted
    ]

    if not accepted:
        raise RuntimeError(
            "No valid initialization pair found meeting "
            "geometric criteria"
        )

    if tracks is not None:
        for evaluation in accepted:
            (
                evaluation.expandable_camera_count,
                evaluation.best_next_image,
                evaluation.best_next_correspondences,
            ) = initial_pair_expansion_support(
                evaluation.image_a,
                evaluation.image_b,
                tracks,
                min_pnp_correspondences,
            )

    return max(
        accepted,
        key=_initial_pair_selection_key,
    )


def select_init_pair(
    pair_geometries: dict[
        tuple[int, int],
        PairGeometry,
    ],
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig | None = None,
    tracks: list[Track] | None = None,
) -> tuple[
    InitialPairEvaluation,
    list[InitialPairEvaluation],
]:
    """Evaluate and select the initial image pair.

    Two separate decisions are performed:

    1. ``evaluate_pair_geometry`` decides whether each pair is
       geometrically reliable.
    2. ``choose_best_initial_pair_evaluation`` decides which accepted
       pair is the best seed for incremental expansion.
    """

    if config is None:
        config = SfMConfig()

    evaluations = [
        evaluate_pair_geometry(
            pair_geometry,
            keypoints,
            K,
            config,
        )
        for pair_geometry
        in pair_geometries.values()
    ]

    best_evaluation = (
        choose_best_initial_pair_evaluation(
            evaluations,
            tracks=tracks,
            min_pnp_correspondences=(
                config.min_pnp_correspondences
            ),
        )
    )

    return (
        best_evaluation,
        evaluations,
    )