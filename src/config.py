"""Configuration dataclasses for Structure-from-Motion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SfMConfig:
    # Random seed
    global_seed: int = 42

    # matching / geometry
    min_verified_matches: int = 30
    essential_ransac_px: float = 3.0
    essential_ransac_iters: int = 2000

    # initialization
    min_init_inliers: int = 100
    min_parallax_deg: float = 1.0
    min_cheirality_ratio: float = 0.75
    homography_ransac_px: float = 3.0
    homography_dominance_ratio: float = 0.95

    # triangulation
    max_triangulation_error_px: float = 4.0

    # PnP
    min_pnp_correspondences: int = 15
    min_pnp_inliers: int = 12
    min_pnp_inlier_ratio: float = 0.30
    max_pnp_rms_px: float = 4.0
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

    def __post_init__(self):
        if self.min_pnp_correspondences < self.min_pnp_inliers:
            raise ValueError(
                f"Invalid PnP configuration: min_pnp_correspondences ({self.min_pnp_correspondences}) "
                f"must be >= min_pnp_inliers ({self.min_pnp_inliers})"
            )
        if not (0.0 <= self.min_pnp_inlier_ratio <= 1.0):
            raise ValueError(
                f"min_pnp_inlier_ratio must be in [0, 1], got {self.min_pnp_inlier_ratio}"
            )
        if self.max_pnp_rms_px <= 0:
            raise ValueError(
                f"max_pnp_rms_px must be positive, got {self.max_pnp_rms_px}"
            )
