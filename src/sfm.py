"""Incremental Structure-from-Motion façade.

This module re-exports the public and pipeline API for backwards compatibility.
The implementation is partitioned into src.pipeline modules.
"""

from .pipeline import (
    _failure_stats,
    _run_bundle_adjustment_on_state,
    _sampson_thresh,
    choose_next_camera,
    collect_bundle_observations,
    compute_reprojection_rms,
    compute_view_graph_components,
    prune_outliers,
    run_sfm,
    triangulate_new_landmarks,
    verify_match_graph,
)

__all__ = [
    "_failure_stats",
    "_run_bundle_adjustment_on_state",
    "_sampson_thresh",
    "choose_next_camera",
    "collect_bundle_observations",
    "compute_reprojection_rms",
    "compute_view_graph_components",
    "prune_outliers",
    "run_sfm",
    "triangulate_new_landmarks",
    "verify_match_graph",
]