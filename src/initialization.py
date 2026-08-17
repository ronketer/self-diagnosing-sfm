"""Two-view initialization selection with explicit geometric evaluation and model evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .config import SfMConfig
from .geometry.essential import choose_pose, decompose_essential, ransac_essential, sampson_distance
from .geometry.projection import reproject, to_normalized, triangulation_angles_deg
from .models import PairGeometry
from .triangulation import triangulate_two_view
from .validation.degeneracy import evaluate_homography_support
from .validation.diagnostics import FailureReason


@dataclass
class InitialPairEvaluation:
    """Detailed geometric evaluation of an initialization pair candidate."""
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

    accepted: bool = False
    rejection_reason: FailureReason | str | None = None
    score: float = 0.0

    E: np.ndarray | None = None
    R: np.ndarray | None = None
    t: np.ndarray | None = None
    inlier_mask: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair": [self.image_a, self.image_b],
            "n_matches": self.n_matches,
            "n_inliers": self.n_inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "cheirality_ratio": round(self.cheirality_ratio, 4),
            "median_parallax_deg": round(self.median_parallax_deg, 3),
            "median_reprojection_px": round(self.median_reprojection_px, 3),
            "homography_inliers": self.homography_inliers,
            "homography_inlier_ratio": round(self.homography_inlier_ratio, 4),
            "homography_dominance_ratio": round(self.homography_dominance_ratio, 4),
            "is_homography_dominant": self.is_homography_dominant,
            "accepted": self.accepted,
            "rejection_reason": (
                self.rejection_reason.value
                if isinstance(self.rejection_reason, FailureReason)
                else self.rejection_reason
            ),
            "score": round(self.score, 4),
        }


def _sampson_thresh(K: np.ndarray, px: float = 3.0) -> float:
    f = (K[0, 0] + K[1, 1]) / 2.0
    return (px / f) ** 2


def evaluate_pair(
    i: int,
    j: int,
    matches: np.ndarray,
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig | None = None,
) -> InitialPairEvaluation:
    """Evaluate a single raw pair candidate (legacy/fallback API wrapper)."""
    if config is None:
        config = SfMConfig()

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

    x1 = to_normalized(keypoints[i][matches[:, 0]], K)
    x2 = to_normalized(keypoints[j][matches[:, 1]], K)
    thresh = _sampson_thresh(K, px=config.essential_ransac_px)

    pair_seed = int((i * 10007 + j * 997 + config.global_seed) % (2**31 - 1))
    try:
        E, inliers = ransac_essential(
            x1, x2, thresh=thresh, iters=config.essential_ransac_iters, seed=pair_seed
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
            rejection_reason=FailureReason.ESSENTIAL_ESTIMATION_FAILED,
        )

    pg = PairGeometry(
        image_a=i,
        image_b=j,
        raw_matches=matches,
        verified_matches=matches[inliers],
        E=E,
        inlier_mask=inliers,
        n_raw_matches=n_matches,
        n_inliers=int(inliers.sum()),
        inlier_ratio=float(inliers.sum() / max(n_matches, 1)),
    )
    return evaluate_pair_geometry(pg, keypoints, K, config)


def evaluate_pair_geometry(
    pg: PairGeometry,
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig,
) -> InitialPairEvaluation:
    """Evaluate already-estimated PairGeometry against all geometric criteria without duplicate RANSAC."""
    i, j = pg.image_a, pg.image_b
    n_matches = pg.n_raw_matches
    n_inliers = pg.n_inliers
    inlier_ratio = pg.inlier_ratio

    if n_matches < config.min_verified_matches or n_inliers < config.min_init_inliers:
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

    pts_px1 = keypoints[i][pg.verified_matches[:, 0]]
    pts_px2 = keypoints[j][pg.verified_matches[:, 1]]

    # 1. Homography vs Essential Model Evidence
    _, _, h_inliers, h_ratio = evaluate_homography_support(
        pts_px1, pts_px2, thresh_px=config.homography_ransac_px
    )
    h_dominance = float(h_inliers / max(n_inliers, 1))

    # Normalized coordinates
    x1 = to_normalized(pts_px1, K)
    x2 = to_normalized(pts_px2, K)

    # 2. Decompose Essential Matrix & Cheirality Check
    candidates = decompose_essential(pg.E)
    R, t, cheir = choose_pose(candidates, x1, x2)
    cheirality_ratio = float(cheir.sum() / max(len(x1), 1))

    if cheirality_ratio < config.min_cheirality_ratio:
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
            is_homography_dominant=(h_dominance >= config.homography_dominance_ratio),
            accepted=False,
            rejection_reason=FailureReason.POOR_CHEIRALITY,
            score=0.0,
            E=pg.E,
            R=R,
            t=t,
            inlier_mask=pg.inlier_mask,
        )

    # 3. Triangulate inliers passing cheirality
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    X = triangulate_two_view(P1, P2, x1[cheir], x2[cheir])

    # 4. Parallax Angle Check
    C1 = np.zeros(3)
    C2 = -R.T @ t
    angles = triangulation_angles_deg(C1, C2, X)
    median_parallax = float(np.median(angles))

    is_homography_dominant = (
        h_dominance >= config.homography_dominance_ratio
        and median_parallax < config.min_parallax_deg
    )

    if median_parallax < config.min_parallax_deg:
        reason = FailureReason.HOMOGRAPHY_DOMINANT if is_homography_dominant else FailureReason.LOW_PARALLAX
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

    # 5. Pixel Reprojection Check
    proj1 = reproject(K, np.eye(3), np.zeros(3), X)
    proj2 = reproject(K, R, t, X)
    errs1 = np.linalg.norm(proj1 - pts_px1[cheir], axis=1)
    errs2 = np.linalg.norm(proj2 - pts_px2[cheir], axis=1)
    median_reproj = float(np.median(np.maximum(errs1, errs2)))

    if median_reproj > config.max_triangulation_error_px:
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
            rejection_reason=FailureReason.HIGH_REPROJECTION_ERROR,
            score=0.0,
            E=pg.E,
            R=R,
            t=t,
            inlier_mask=pg.inlier_mask,
        )

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


def select_init_pair(
    pair_geometries: dict[tuple[int, int], PairGeometry],
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig | None = None,
) -> tuple[InitialPairEvaluation, list[InitialPairEvaluation]]:
    """Evaluate candidate PairGeometries directly without redundant RANSAC and return the best accepted pair."""
    if config is None:
        config = SfMConfig()

    evaluations: list[InitialPairEvaluation] = []
    best_eval: InitialPairEvaluation | None = None

    for pg in pair_geometries.values():
        ev = evaluate_pair_geometry(pg, keypoints, K, config)
        evaluations.append(ev)
        if ev.accepted:
            if best_eval is None or ev.score > best_eval.score:
                best_eval = ev

    if best_eval is None:
        raise RuntimeError("No valid initialization pair found meeting geometric criteria")

    return best_eval, evaluations
