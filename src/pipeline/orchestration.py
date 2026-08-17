"""High-level incremental Structure-from-Motion stage sequencing."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..config import SfMConfig
from ..models import CameraIntrinsics, Reconstruction
from ..tracks import TrackManager
from ..validation.diagnostics import ReconstructionDiagnostics
from .finalization import finalize_reconstruction
from .initialization_stage import initialize_reconstruction
from .registration import run_incremental_registration
from .verification import verify_match_graph


def run_sfm(
    keypoints: list[np.ndarray],
    match_graph: dict[
        tuple[int, int],
        np.ndarray,
    ],
    K: np.ndarray | CameraIntrinsics,
    config: SfMConfig | None = None,
    ba_every: int = 3,
    use_ba: bool = True,
    use_geometric_verification: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[
    Reconstruction,
    TrackManager,
    dict[str, Any],
]:
    """Run robust incremental SfM and return reconstruction, tracks, and stats."""
    if config is None:
        config = SfMConfig(ba_every_n_cameras=ba_every)

    intrinsics = (
        K
        if isinstance(K, CameraIntrinsics)
        else CameraIntrinsics.from_matrix(K)
    )
    K_mat = intrinsics.K

    raw_pairs = len(match_graph)
    raw_matches = sum(len(matches) for matches in match_graph.values())

    # ------------------------------------------------------------------
    # 1. Pairwise geometric verification.
    # ------------------------------------------------------------------
    if use_geometric_verification:
        (
            pair_geometries,
            verified_match_graph,
        ) = verify_match_graph(
            match_graph,
            keypoints,
            intrinsics,
            config,
            log,
        )
    else:
        log("geometric verification disabled (ablation)")
        pair_geometries = {}
        verified_match_graph = {
            pair: np.asarray(matches, dtype=np.int64).reshape(-1, 2)
            for pair, matches in match_graph.items()
        }

    # ------------------------------------------------------------------
    # 2. Conflict-aware feature tracks.
    # ------------------------------------------------------------------
    track_manager = TrackManager(
        (
            pair_geometries
            if use_geometric_verification
            else verified_match_graph
        ),
        keypoints=keypoints,
        K=K_mat,
    )
    log(f"tracks: {len(track_manager.tracks)}")

    diagnostics = ReconstructionDiagnostics()
    diagnostics.track_conflict_rejected_edges = (
        track_manager.n_rejected_edges
    )

    # ------------------------------------------------------------------
    # 3. Two-view initialization.
    # ------------------------------------------------------------------
    init_result = initialize_reconstruction(
        intrinsics=intrinsics,
        keypoints=keypoints,
        match_graph=match_graph,
        pair_geometries=pair_geometries,
        verified_match_graph=verified_match_graph,
        track_manager=track_manager,
        diagnostics=diagnostics,
        config=config,
        raw_pairs=raw_pairs,
        raw_matches=raw_matches,
        use_ba=use_ba,
        use_geometric_verification=use_geometric_verification,
        log=log,
    )

    if not init_result.is_success or init_result.state is None:
        reconstruction = Reconstruction(intrinsics=intrinsics)
        return (
            reconstruction,
            track_manager,
            init_result.failure_stats or {},
        )

    state = init_result.state
    image_0 = init_result.image_0
    image_1 = init_result.image_1
    assert image_0 is not None and image_1 is not None

    # ------------------------------------------------------------------
    # 4. Incremental camera registration.
    # ------------------------------------------------------------------
    failed_pnp_images = run_incremental_registration(
        state,
        use_ba=use_ba,
        log=log,
    )

    # ------------------------------------------------------------------
    # 5. Finalization (final optimization, camera statuses, stats, summary).
    # ------------------------------------------------------------------
    finalize_reconstruction(
        state,
        failed_pnp_images=failed_pnp_images,
        pair_geometries=pair_geometries,
        match_graph=match_graph,
        verified_match_graph=verified_match_graph,
        image_0=image_0,
        image_1=image_1,
        raw_pairs=raw_pairs,
        raw_matches=raw_matches,
        use_ba=use_ba,
        use_geometric_verification=use_geometric_verification,
        log=log,
    )

    return (
        state.reconstruction,
        state.track_manager,
        state.stats,
    )
