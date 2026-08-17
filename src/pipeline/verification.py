"""Pairwise geometric verification for Structure-from-Motion."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..config import SfMConfig
from ..geometry.essential import (
    ransac_essential,
    sampson_distance,
    sampson_threshold_from_pixels,
)
from ..geometry.projection import to_normalized
from ..models import CameraIntrinsics, PairGeometry


def verify_match_graph(
    match_graph: dict[
        tuple[int, int],
        np.ndarray,
    ],
    keypoints: list[np.ndarray],
    intrinsics: CameraIntrinsics | np.ndarray,
    config: SfMConfig,
    log: Callable[[str], None] = print,
) -> tuple[
    dict[tuple[int, int], PairGeometry],
    dict[tuple[int, int], np.ndarray],
]:
    """Estimate Essential geometry exactly once for each candidate pair."""
    K = (
        intrinsics.K
        if isinstance(intrinsics, CameraIntrinsics)
        else np.asarray(intrinsics, dtype=float)
    )

    pair_geometries: dict[
        tuple[int, int],
        PairGeometry,
    ] = {}

    verified_matches: dict[
        tuple[int, int],
        np.ndarray,
    ] = {}

    threshold = sampson_threshold_from_pixels(
        K,
        px=config.essential_ransac_px,
    )

    for (
        image_a,
        image_b,
    ), matches_in in sorted(match_graph.items()):
        matches = np.asarray(
            matches_in,
            dtype=np.int64,
        ).reshape(-1, 2)

        if len(matches) < config.min_verified_matches:
            continue

        x_a = to_normalized(
            keypoints[image_a][matches[:, 0]],
            K,
        )
        x_b = to_normalized(
            keypoints[image_b][matches[:, 1]],
            K,
        )

        pair_seed = int(
            (
                image_a * 10007
                + image_b * 997
                + config.global_seed
            )
            % (2**31 - 1)
        )

        try:
            E, inlier_mask = ransac_essential(
                x_a,
                x_b,
                thresh=threshold,
                iters=config.essential_ransac_iters,
                seed=pair_seed,
            )
        except RuntimeError:
            continue

        n_inliers = int(inlier_mask.sum())
        if n_inliers < config.min_verified_matches:
            continue

        pair_matches = matches[inlier_mask]
        errors = sampson_distance(
            E,
            x_a[inlier_mask],
            x_b[inlier_mask],
        )

        geometry = PairGeometry(
            image_a=image_a,
            image_b=image_b,
            raw_matches=matches,
            verified_matches=pair_matches,
            E=E,
            inlier_mask=inlier_mask,
            n_raw_matches=len(matches),
            n_inliers=n_inliers,
            inlier_ratio=float(
                n_inliers / len(matches)
            ),
            median_sampson_error=(
                float(np.median(errors))
                if len(errors)
                else None
            ),
        )

        pair_geometries[(image_a, image_b)] = geometry
        verified_matches[(image_a, image_b)] = pair_matches

    n_raw = sum(
        len(matches)
        for matches in match_graph.values()
    )
    n_verified = sum(
        len(matches)
        for matches in verified_matches.values()
    )

    log(
        f"geometric verification: "
        f"{n_raw} -> {n_verified} matches, "
        f"{len(match_graph)} -> "
        f"{len(verified_matches)} pairs"
    )

    return (
        pair_geometries,
        verified_matches,
    )
