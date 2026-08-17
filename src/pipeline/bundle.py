"""Bundle adjustment numerical boundary execution for SfM pipeline."""

from __future__ import annotations

from ..bundle_adjustment import run_bundle_adjustment
from ..models import ReconstructionState
from .reconstruction_ops import collect_bundle_observations


def _run_bundle_adjustment_on_state(
    state: ReconstructionState,
) -> tuple[
    float,
    float,
    int,
]:
    """Numerical BA boundary: typed state -> arrays -> typed state."""
    observations = collect_bundle_observations(state)
    reconstruction = state.reconstruction

    (
        poses,
        points,
        rms_before,
        rms_after,
    ) = run_bundle_adjustment(
        reconstruction.intrinsics,
        reconstruction.poses,
        reconstruction.points3d,
        observations,
        anchor_camera=state.anchor_camera,
        scale_camera=state.scale_camera,
        initial_baseline=state.initial_baseline,
        config=state.config,
    )

    reconstruction.apply_optimization(
        poses,
        points,
    )

    return (
        rms_before,
        rms_after,
        len(observations),
    )
