"""Incremental Structure-from-Motion façade.

This module re-exports the public and pipeline API for backwards compatibility.
The implementation is partitioned into src.pipeline modules.
"""

from .pipeline import (
    _failure_stats,
    _run_bundle_adjustment_on_state,
    _sampson_thresh,
    build_final_stats,
    choose_next_camera,
    collect_bundle_observations,
    compute_reprojection_rms,
    compute_view_graph_components,
    finalize_camera_statuses,
    finalize_reconstruction,
    log_final_summary,
    prune_outliers,
    run_final_optimization,
    run_incremental_registration,
    run_sfm,
    triangulate_new_landmarks,
    verify_match_graph,
)

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