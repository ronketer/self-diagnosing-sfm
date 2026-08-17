"""Configuration dataclasses for Structure-from-Motion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SfMConfig:
    # matching / geometry
    min_verified_matches: int = 30
    essential_ransac_px: float = 3.0
    essential_ransac_iters: int = 2000

    # initialization
    min_init_inliers: int = 100
    min_parallax_deg: float = 1.0
    min_cheirality_ratio: float = 0.75

    # triangulation
    max_triangulation_error_px: float = 4.0

    # PnP
    min_pnp_correspondences: int = 15
    min_pnp_inliers: int = 20
    pnp_reprojection_error_px: float = 4.0
    pnp_confidence: float = 0.999
    pnp_iterations: int = 2000

    # bundle adjustment
    ba_every_n_cameras: int = 3
    ba_robust_scale_px: float = 2.0
    scale_constraint_weight: float = 100.0

    # landmark filtering / pruning
    max_landmark_error_px: float = 4.0
    post_ba_prune_error_px: float = 4.0
    final_prune_error_px: float = 2.0
