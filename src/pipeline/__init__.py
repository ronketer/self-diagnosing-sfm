"""Structure-from-Motion incremental reconstruction pipeline."""

from .bundle import _run_bundle_adjustment_on_state
from .diagnostics import _failure_stats, compute_view_graph_components
from .finalization import (
    build_final_stats,
    finalize_camera_statuses,
    finalize_reconstruction,
    log_final_summary,
    run_final_optimization,
)
from .orchestration import run_sfm
from .reconstruction_ops import (
    collect_bundle_observations,
    compute_reprojection_rms,
    prune_outliers,
    triangulate_new_landmarks,
)
from .registration import choose_next_camera, run_incremental_registration
from .verification import _sampson_thresh, verify_match_graph

__all__ = [
    "_failure_stats",
    "_run_bundle_adjustment_on_state",
    "_sampson_thresh",
    "build_final_stats",
    "choose_next_camera",
    "collect_bundle_observations",
    "compute_reprojection_rms",
    "compute_view_graph_components",
    "finalize_camera_statuses",
    "finalize_reconstruction",
    "log_final_summary",
    "prune_outliers",
    "run_final_optimization",
    "run_incremental_registration",
    "run_sfm",
    "triangulate_new_landmarks",
    "verify_match_graph",
]
