"""Two-view initialization selection with explicit geometric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from .config import SfMConfig
from .geometry.essential import choose_pose, decompose_essential, ransac_essential
from .geometry.projection import reproject, to_normalized, triangulation_angles_deg
from .triangulation import triangulate_two_view
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

    accepted: bool
    rejection_reason: FailureReason | str | None
    score: float

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
            "accepted": self.accepted,
            "rejection_reason": (
                self.rejection_reason.value
                if isinstance(self.rejection_reason, FailureReason)
                else self.rejection_reason
            ),
            "score": round(self.score, 4),
        }


def _sampson_thresh(K: np.ndarray, px: float = 3.0) -> float:
    """Squared Sampson threshold in normalized coords for pixel tolerance."""
    f = (K[0, 0] + K[1, 1]) / 2.0
    return (px / f) ** 2


def evaluate_pair(
    i: int,
    j: int,
    matches: np.ndarray,
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig,
) -> InitialPairEvaluation:
    """Evaluate a single pair candidate against all geometric criteria."""
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
            score=0.0,
        )

    x1 = to_normalized(keypoints[i][matches[:, 0]], K)
    x2 = to_normalized(keypoints[j][matches[:, 1]], K)
    thresh = _sampson_thresh(K, px=config.essential_ransac_px)

    try:
        E, inl = ransac_essential(x1, x2, thresh=thresh, iters=config.essential_ransac_iters)
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
            score=0.0,
        )

    n_inliers = int(inl.sum())
    inlier_ratio = float(n_inliers / n_matches)
    if n_inliers < config.min_init_inliers:
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
            E=E,
            inlier_mask=inl,
        )

    candidates = decompose_essential(E)
    R, t, cheir = choose_pose(candidates, x1[inl], x2[inl])
    cheirality_ratio = float(cheir.sum() / max(n_inliers, 1))

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
            accepted=False,
            rejection_reason=FailureReason.POOR_CHEIRALITY,
            score=0.0,
            E=E,
            R=R,
            t=t,
            inlier_mask=inl,
        )

    # Triangulate sample inliers passing cheirality
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R, t.reshape(3, 1)])
    X = triangulate_two_view(P1, P2, x1[inl][cheir], x2[inl][cheir])

    C1 = np.zeros(3)
    C2 = -R.T @ t
    angles = triangulation_angles_deg(C1, C2, X)
    median_parallax = float(np.median(angles))

    if median_parallax < config.min_parallax_deg:
        return InitialPairEvaluation(
            image_a=i,
            image_b=j,
            n_matches=n_matches,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            cheirality_ratio=cheirality_ratio,
            median_parallax_deg=median_parallax,
            median_reprojection_px=0.0,
            accepted=False,
            rejection_reason=FailureReason.LOW_PARALLAX,
            score=0.0,
            E=E,
            R=R,
            t=t,
            inlier_mask=inl,
        )

    # Check reprojections
    pts_px1 = keypoints[i][matches[inl][cheir, 0]]
    pts_px2 = keypoints[j][matches[inl][cheir, 1]]
    proj1 = reproject(K, np.eye(3), np.zeros(3), X)
    proj2 = reproject(K, R, t, X)
    errs1 = np.linalg.norm(proj1 - pts_px1, axis=1)
    errs2 = np.linalg.norm(proj2 - pts_px2, axis=1)
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
            accepted=False,
            rejection_reason=FailureReason.HIGH_REPROJECTION_ERROR,
            score=0.0,
            E=E,
            R=R,
            t=t,
            inlier_mask=inl,
        )

    # Score prioritizes triangulation angle among well-supported candidates
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
        accepted=True,
        rejection_reason=None,
        score=score,
        E=E,
        R=R,
        t=t,
        inlier_mask=inl,
    )


def select_init_pair(
    match_graph: dict[tuple[int, int], np.ndarray],
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig | None = None,
) -> tuple[InitialPairEvaluation, list[InitialPairEvaluation]]:
    """Evaluate all candidate pairs and return the best accepted pair along with all evaluations."""
    if config is None:
        config = SfMConfig()

    evaluations: list[InitialPairEvaluation] = []
    best_eval: InitialPairEvaluation | None = None

    for (i, j), matches in match_graph.items():
        ev = evaluate_pair(i, j, matches, keypoints, K, config)
        evaluations.append(ev)
        if ev.accepted:
            if best_eval is None or ev.score > best_eval.score:
                best_eval = ev

    if best_eval is None:
        raise RuntimeError("No valid initialization pair found meeting geometric criteria")

    return best_eval, evaluations
