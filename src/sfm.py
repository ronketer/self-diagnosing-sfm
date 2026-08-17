"""Incremental Structure-from-Motion pipeline with explicit validation and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np

from .bundle_adjustment import run_bundle_adjustment
from .config import SfMConfig
from .geometry.essential import ransac_essential
from .geometry.projection import reproject, to_normalized
from .initialization import InitialPairEvaluation, select_init_pair
from .models import CameraIntrinsics, CameraPose, Reconstruction
from .pnp import PnPResult, estimate_camera_pose
from .tracks import TrackManager
from .triangulation import point_depths, triangulate_track
from .validation.diagnostics import (
    DiagnosticEvent,
    FailureReason,
    ReconstructionDiagnostics,
)


@dataclass
class ReconstructionState:
    """Encapsulates the evolving state of an incremental SfM reconstruction."""
    recon: Reconstruction
    tracks: TrackManager
    keypoints: list[np.ndarray]
    config: SfMConfig
    diagnostics: ReconstructionDiagnostics = field(default_factory=ReconstructionDiagnostics)
    registered_images: set[int] = field(default_factory=set)
    registration_order: list[int] = field(default_factory=list)
    anchor_camera: int = 0
    scale_camera: int = 1
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_images(self) -> int:
        return len(self.keypoints)

    def has_unregistered_images(self) -> bool:
        return len(self.registered_images) < self.n_images

    def add_camera(self, image_id: int, pose: CameraPose) -> None:
        self.recon.set_pose(image_id, pose)
        self.registered_images.add(image_id)
        self.registration_order.append(image_id)

    def get_observations(self) -> list[tuple[int, int, np.ndarray]]:
        """(img, point3d_id, xy_pixel) for all triangulated tracks in registered views."""
        obs = []
        for tr in self.tracks.tracks:
            if tr.point3d_id is None:
                continue
            for img, kp in tr.obs.items():
                if img in self.registered_images:
                    obs.append((img, tr.point3d_id, self.keypoints[img][kp]))
        return obs

    def compute_reprojection_rms(self) -> float | None:
        obs = self.get_observations()
        if not obs:
            return None
        sq_errs = []
        for img, pt_id, xy in obs:
            R, t = self.recon.poses[img]
            proj = reproject(self.recon.K, R, t, self.recon.points3d[pt_id].reshape(1, 3))[0]
            sq_errs.append(float(np.sum((proj - xy) ** 2)))
        return float(np.sqrt(np.mean(sq_errs)))


def _sampson_thresh(K: np.ndarray, px: float = 3.0) -> float:
    f = (K[0, 0] + K[1, 1]) / 2.0
    return (px / f) ** 2


def verify_match_graph(
    match_graph: dict[tuple[int, int], np.ndarray],
    keypoints: list[np.ndarray],
    K: np.ndarray,
    config: SfMConfig,
    log: Callable[[str], None] = print,
) -> dict[tuple[int, int], np.ndarray]:
    """Epipolar geometric verification: keep only essential-consistent matches per pair."""
    verified = {}
    thresh = _sampson_thresh(K, px=config.essential_ransac_px)
    for (i, j), matches in match_graph.items():
        if len(matches) < config.min_verified_matches:
            continue
        x1 = to_normalized(keypoints[i][matches[:, 0]], K)
        x2 = to_normalized(keypoints[j][matches[:, 1]], K)
        try:
            _, inl = ransac_essential(
                x1, x2, thresh=thresh, iters=min(1000, config.essential_ransac_iters)
            )
        except RuntimeError:
            continue
        if inl.sum() >= config.min_verified_matches:
            verified[(i, j)] = matches[inl]
    n_in = sum(len(m) for m in match_graph.values())
    n_out = sum(len(m) for m in verified.values())
    log(
        f"geometric verification: {n_in} -> {n_out} matches, "
        f"{len(match_graph)} -> {len(verified)} pairs"
    )
    return verified


def triangulate_new_landmarks(
    state: ReconstructionState,
    log: Callable[[str], None] | None = None,
) -> int:
    """Triangulate all newly triangulatable tracks among currently registered views."""
    tm = state.tracks
    recon = state.recon
    keypoints = state.keypoints
    config = state.config

    new_pts = []
    new_track_indices = []

    for ti in tm.triangulatable_tracks(state.registered_images):
        tr = tm.tracks[ti]
        res = triangulate_track(tr.obs, recon.poses, keypoints, recon.K, config)
        if res.accepted and res.xyz is not None:
            new_pts.append(res.xyz)
            new_track_indices.append(ti)

    if new_pts:
        base = len(recon.points3d)
        state.recon.points3d = (
            np.vstack([recon.points3d, np.array(new_pts)])
            if len(recon.points3d) > 0
            else np.array(new_pts)
        )
        for k, ti in enumerate(new_track_indices):
            tm.tracks[ti].point3d_id = base + k

    return len(new_pts)


def prune_outliers(
    state: ReconstructionState,
    max_err: float = 4.0,
) -> int:
    """Drop 3D landmarks whose mean reprojection error exceeds max_err, then compact points."""
    recon = state.recon
    tm = state.tracks
    keypoints = state.keypoints
    K = recon.K
    registered = state.registered_images

    if len(recon.points3d) == 0:
        return 0

    keep = np.ones(len(recon.points3d), dtype=bool)
    for tr in tm.tracks:
        if tr.point3d_id is None:
            continue
        errs = []
        X = recon.points3d[tr.point3d_id].reshape(1, 3)
        for img, kp in tr.obs.items():
            if img in registered:
                R, t = recon.poses[img]
                proj = reproject(K, R, t, X)[0]
                errs.append(np.linalg.norm(proj - keypoints[img][kp]))
        if errs and np.mean(errs) > max_err:
            keep[tr.point3d_id] = False

    remap = -np.ones(len(keep), dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    recon.points3d = recon.points3d[keep]

    removed = 0
    for tr in tm.tracks:
        if tr.point3d_id is not None:
            nid = remap[tr.point3d_id]
            tr.point3d_id = int(nid) if nid >= 0 else None
            removed += nid < 0

    return int((~keep).sum())


def choose_next_camera(
    state: ReconstructionState,
    deferred: set[int],
) -> tuple[int | None, int]:
    """Find the unregistered camera sharing the most 2D-3D correspondences."""
    best_img, best_n = None, 0
    for img in range(state.n_images):
        if img in state.registered_images or img in deferred:
            continue
        p2, _, _ = state.tracks.correspondences_2d3d(
            img, state.keypoints, state.recon.points3d
        )
        if len(p2) > best_n:
            best_img, best_n = img, len(p2)
    return best_img, best_n


def run_sfm(
    keypoints: list[np.ndarray],
    match_graph: dict,
    K: np.ndarray | CameraIntrinsics,
    config: SfMConfig | None = None,
    ba_every: int = 3,
    use_ba: bool = True,
    use_geometric_verification: bool = True,
    log: Callable[[str], None] = print,
) -> tuple[Reconstruction, TrackManager, dict]:
    """Execute the full incremental Structure-from-Motion pipeline."""
    if config is None:
        config = SfMConfig(ba_every_n_cameras=ba_every)

    if isinstance(K, CameraIntrinsics):
        K_mat = K.K
    else:
        K_mat = np.asarray(K, dtype=float)

    raw_pairs = len(match_graph)
    raw_matches = sum(len(matches) for matches in match_graph.values())

    # Step 1: Geometric Verification
    if use_geometric_verification:
        match_graph = verify_match_graph(match_graph, keypoints, K_mat, config, log)
    else:
        log("geometric verification disabled (ablation)")

    # Step 2: Track Construction
    tm = TrackManager(match_graph)
    log(f"tracks: {len(tm.tracks)}")

    # Step 3: Two-View Initialization
    best_init, init_evals = select_init_pair(match_graph, keypoints, K_mat, config)
    i0, j0 = best_init.image_a, best_init.image_b
    log(f"init pair: ({i0}, {j0}) with {best_init.n_inliers} inliers")

    recon = Reconstruction(K=K_mat)
    recon.set_pose(i0, CameraPose(R=np.eye(3), t=np.zeros(3)))
    recon.set_pose(j0, CameraPose(R=best_init.R, t=best_init.t))

    diagnostics = ReconstructionDiagnostics()
    for ev in init_evals:
        diagnostics.record_pair_evaluation(ev.to_dict())

    state = ReconstructionState(
        recon=recon,
        tracks=tm,
        keypoints=keypoints,
        config=config,
        diagnostics=diagnostics,
        registered_images={i0, j0},
        registration_order=[i0, j0],
        anchor_camera=i0,
        scale_camera=j0,
    )

    # Initial Triangulation
    n_new = triangulate_new_landmarks(state, log)
    log(f"initial triangulation: {n_new} points")

    stats: dict[str, Any] = {
        "registration_order": [i0, j0],
        "n_images": len(keypoints),
        "use_ba": use_ba,
        "use_geometric_verification": use_geometric_verification,
        "raw_match_pairs": raw_pairs,
        "raw_matches": raw_matches,
        "verified_match_pairs": len(match_graph),
        "verified_matches": sum(len(matches) for matches in match_graph.values()),
        "ba_history": [],
        "diagnostics": diagnostics,
    }

    since_ba = 0
    deferred: set[int] = set()

    # Step 4: Incremental Registration Loop
    while state.has_unregistered_images():
        best_img, n_corr = choose_next_camera(state, deferred)

        if best_img is None or n_corr < config.min_pnp_correspondences:
            log(f"stopping: no registrable image left ({n_corr} correspondences)")
            break

        p2, p3, _ = tm.correspondences_2d3d(best_img, keypoints, recon.points3d)
        pnp_result = estimate_camera_pose(p3, p2, K_mat, config)

        if not pnp_result.accepted or pnp_result.pose is None:
            n_inl = len(pnp_result.inlier_indices)
            log(f"image {best_img}: PnP rejected ({n_inl}/{len(p2)} inliers); deferring")
            deferred.add(best_img)
            event = DiagnosticEvent(
                stage="CAMERA_REGISTRATION",
                entity=f"image:{best_img}",
                status="DEFERRED",
                reason=pnp_result.reason or FailureReason.INSUFFICIENT_PNP_INLIERS,
                metrics={
                    "correspondences": len(p2),
                    "inliers": n_inl,
                    "inlier_ratio": float(n_inl / max(len(p2), 1)),
                },
            )
            diagnostics.record(event)
            continue

        # Register Camera
        state.add_camera(best_img, pnp_result.pose)
        deferred.clear()  # Map has expanded; retrying previously deferred images is now viable
        stats["registration_order"].append(best_img)

        event = DiagnosticEvent(
            stage="CAMERA_REGISTRATION",
            entity=f"image:{best_img}",
            status="REGISTERED",
            reason=None,
            metrics={
                "correspondences": len(p2),
                "inliers": len(pnp_result.inlier_indices),
                "inlier_ratio": float(len(pnp_result.inlier_indices) / max(len(p2), 1)),
                "reprojection_rms_px": pnp_result.reprojection_rms_px or 0.0,
            },
        )
        diagnostics.record(event)

        n_new = triangulate_new_landmarks(state, log)
        log(
            f"registered image {best_img}: {len(pnp_result.inlier_indices)}/{len(p2)} PnP inliers, "
            f"+{n_new} points (total {len(recon.points3d)})"
        )

        since_ba += 1
        if use_ba and since_ba >= config.ba_every_n_cameras:
            obs = state.get_observations()
            recon.poses, recon.points3d, rb, ra = run_bundle_adjustment(
                K_mat,
                recon.poses,
                recon.points3d,
                obs,
                anchor_camera=state.anchor_camera,
                scale_camera=state.scale_camera,
                config=config,
            )
            n_pruned = prune_outliers(state, max_err=config.post_ba_prune_error_px)
            log(f"  BA: rms {rb:.3f} -> {ra:.3f} px ({len(obs)} obs), pruned {n_pruned}")
            stats["ba_history"].append(
                {
                    "stage": "periodic",
                    "n_registered": len(state.registered_images),
                    "n_observations": len(obs),
                    "rms_before_px": rb,
                    "rms_after_px": ra,
                    "n_pruned": n_pruned,
                }
            )
            since_ba = 0

    # Step 5: Final Optimization & Outlier Filtering
    if use_ba:
        obs = state.get_observations()
        recon.poses, recon.points3d, rb, ra = run_bundle_adjustment(
            K_mat,
            recon.poses,
            recon.points3d,
            obs,
            anchor_camera=state.anchor_camera,
            scale_camera=state.scale_camera,
            config=config,
        )
        log(f"final BA: rms {rb:.3f} -> {ra:.3f} px")
        stats["rms_before_final_ba"] = rb
        stats["ba_history"].append(
            {
                "stage": "final",
                "n_registered": len(state.registered_images),
                "n_observations": len(obs),
                "rms_before_px": rb,
                "rms_after_px": ra,
                "n_pruned": 0,
            }
        )

        n_pruned = prune_outliers(state, max_err=config.final_prune_error_px)
        log(f"pruned {n_pruned} outlier points")
        obs = state.get_observations()
        recon.poses, recon.points3d, rb, ra = run_bundle_adjustment(
            K_mat,
            recon.poses,
            recon.points3d,
            obs,
            anchor_camera=state.anchor_camera,
            scale_camera=state.scale_camera,
            config=config,
        )
        log(f"post-prune BA: rms {rb:.3f} -> {ra:.3f} px")
        stats["rms_final"] = ra
        stats["ba_history"].append(
            {
                "stage": "post-prune",
                "n_registered": len(state.registered_images),
                "n_observations": len(obs),
                "rms_before_px": rb,
                "rms_after_px": ra,
                "n_pruned": n_pruned,
            }
        )

    obs = state.get_observations()
    if not use_ba:
        stats["rms_final"] = state.compute_reprojection_rms()

    stats["n_registered"] = len(state.registered_images)
    stats["n_points"] = len(recon.points3d)
    stats["n_observations"] = len(obs)
    stats["mean_track_length"] = len(obs) / max(len(recon.points3d), 1)

    diagnostics.summary_stats = {
        "initial_pair": [i0, j0],
        "registered_images": stats["n_registered"],
        "total_images": stats["n_images"],
        "landmarks": stats["n_points"],
        "observations": stats["n_observations"],
        "mean_track_length": stats["mean_track_length"],
        "final_reprojection_rms_px": stats.get("rms_final"),
    }

    return recon, tm, stats
