"""Incremental structure-from-motion pipeline."""

from dataclasses import dataclass, field

import numpy as np

from .bundle_adjustment import run_bundle_adjustment
from .geometry import (
    choose_pose,
    decompose_essential,
    ransac_essential,
    to_normalized,
)
from .pnp import ransac_pnp
from .tracks import TrackManager
from .triangulation import (
    point_depths,
    reproject,
    triangulate_multi_view,
    triangulation_angles_deg,
)


@dataclass
class Reconstruction:
    K: np.ndarray
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    points3d: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def camera_center(self, i: int) -> np.ndarray:
        R, t = self.poses[i]
        return -R.T @ t


def _observations(tm: TrackManager, keypoints, registered) -> list:
    """(img, point3d_id, xy) for all triangulated tracks in registered views."""
    obs = []
    for tr in tm.tracks:
        if tr.point3d_id is None:
            continue
        for img, kp in tr.obs.items():
            if img in registered:
                obs.append((img, tr.point3d_id, keypoints[img][kp]))
    return obs


def _reprojection_rms(recon: Reconstruction, observations: list) -> float | None:
    """Root-mean-square 2D reprojection distance over observations."""
    if not observations:
        return None
    squared_errors = []
    for img, point_id, xy in observations:
        R, t = recon.poses[img]
        projected = reproject(
            recon.K, R, t, recon.points3d[point_id].reshape(1, 3)
        )[0]
        squared_errors.append(float(np.sum((projected - xy) ** 2)))
    return float(np.sqrt(np.mean(squared_errors)))


def select_init_pair(
    match_graph, keypoints, K, min_inliers: int = 100
) -> tuple[tuple[int, int], np.ndarray, tuple]:
    """Pick the pair maximizing median triangulation angle among well-matched
    pairs. Returns ((i, j), inlier_mask, (E, R, t))."""
    best = None
    for (i, j), matches in match_graph.items():
        if len(matches) < min_inliers:
            continue
        x1 = to_normalized(keypoints[i][matches[:, 0]], K)
        x2 = to_normalized(keypoints[j][matches[:, 1]], K)
        try:
            E, inl = ransac_essential(x1, x2, thresh=_sampson_thresh(K))
        except RuntimeError:
            continue
        if inl.sum() < min_inliers:
            continue
        R, t, cheir = choose_pose(decompose_essential(E), x1[inl], x2[inl])
        from .triangulation import triangulate_two_view

        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t.reshape(3, 1)])
        X = triangulate_two_view(P1, P2, x1[inl][cheir], x2[inl][cheir])
        C2 = -R.T @ t
        angles = triangulation_angles_deg(np.zeros(3), C2, X)
        score = np.median(angles)
        if best is None or score > best[0]:
            best = (score, (i, j), inl, (E, R, t))
    if best is None:
        raise RuntimeError("No valid initialization pair found")
    _, pair, inl, ERt = best
    return pair, inl, ERt


def _try_triangulate(tm: TrackManager, recon: Reconstruction, keypoints, registered):
    """Triangulate all completable tracks; filter by depth/reproj/angle."""
    K = recon.K
    new_pts = []
    new_track_ids = []
    for ti in tm.triangulatable_tracks(registered):
        tr = tm.tracks[ti]
        views = [img for img in tr.obs if img in registered]
        P_list, x_list = [], []
        for img in views:
            R, t = recon.poses[img]
            P_list.append(np.hstack([R, t.reshape(3, 1)]))
            x_list.append(to_normalized(keypoints[img][tr.obs[img]].reshape(1, 2), K)[0])
        X = triangulate_multi_view(P_list, x_list)

        ok = True
        for img in views:
            R, t = recon.poses[img]
            if point_depths(R, t, X.reshape(1, 3))[0] <= 0:
                ok = False
                break
            err = np.linalg.norm(
                reproject(K, R, t, X.reshape(1, 3))[0] - keypoints[img][tr.obs[img]]
            )
            if err > 4.0:
                ok = False
                break
        if ok and len(views) >= 2:
            C1 = recon.camera_center(views[0])
            C2 = recon.camera_center(views[-1])
            ang = triangulation_angles_deg(C1, C2, X.reshape(1, 3))[0]
            if ang < 1.0:
                ok = False
        if ok:
            new_track_ids.append(ti)
            new_pts.append(X)

    if new_pts:
        base = len(recon.points3d)
        recon.points3d = np.vstack([recon.points3d, np.array(new_pts)])
        for k, ti in enumerate(new_track_ids):
            tm.tracks[ti].point3d_id = base + k
    return len(new_pts)


def prune_outliers(tm: TrackManager, recon: Reconstruction, keypoints, registered, max_err=2.0):
    """Drop 3D points whose mean reprojection error exceeds max_err, then
    compact the points array."""
    K = recon.K
    keep = np.ones(len(recon.points3d), dtype=bool)
    for tr in tm.tracks:
        if tr.point3d_id is None:
            continue
        errs = []
        X = recon.points3d[tr.point3d_id].reshape(1, 3)
        for img, kp in tr.obs.items():
            if img in registered:
                R, t = recon.poses[img]
                errs.append(np.linalg.norm(reproject(K, R, t, X)[0] - keypoints[img][kp]))
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


def _sampson_thresh(K: np.ndarray, px: float = 3.0) -> float:
    """Squared Sampson threshold in normalized coords for a pixel tolerance.
    A fixed normalized threshold would implicitly scale with focal length."""
    f = (K[0, 0] + K[1, 1]) / 2
    return (px / f) ** 2


def verify_match_graph(match_graph, keypoints, K, log=print) -> dict:
    """Geometric verification: keep only epipolar-consistent matches per pair.
    Cuts wrong matches (repeated structure) before they contaminate tracks."""
    verified = {}
    thresh = _sampson_thresh(K)
    for (i, j), matches in match_graph.items():
        x1 = to_normalized(keypoints[i][matches[:, 0]], K)
        x2 = to_normalized(keypoints[j][matches[:, 1]], K)
        try:
            _, inl = ransac_essential(x1, x2, thresh=thresh, iters=1000)
        except RuntimeError:
            continue
        if inl.sum() >= 30:
            verified[(i, j)] = matches[inl]
    n_in = sum(len(m) for m in match_graph.values())
    n_out = sum(len(m) for m in verified.values())
    log(f"geometric verification: {n_in} -> {n_out} matches, "
        f"{len(match_graph)} -> {len(verified)} pairs")
    return verified


def run_sfm(
    keypoints: list[np.ndarray],
    match_graph: dict,
    K: np.ndarray,
    ba_every: int = 3,
    use_ba: bool = True,
    use_geometric_verification: bool = True,
    log=print,
) -> tuple[Reconstruction, TrackManager, dict]:
    raw_pairs = len(match_graph)
    raw_matches = sum(len(matches) for matches in match_graph.values())
    if use_geometric_verification:
        match_graph = verify_match_graph(match_graph, keypoints, K, log)
    else:
        log("geometric verification disabled (ablation)")
    tm = TrackManager(match_graph)
    log(f"tracks: {len(tm.tracks)}")

    (i0, j0), inl, (E, R, t) = select_init_pair(match_graph, keypoints, K)
    log(f"init pair: ({i0}, {j0}) with {inl.sum()} inliers")

    recon = Reconstruction(K=K)
    recon.poses[i0] = (np.eye(3), np.zeros(3))
    recon.poses[j0] = (R, t)
    registered = {i0, j0}

    n_new = _try_triangulate(tm, recon, keypoints, registered)
    log(f"initial triangulation: {n_new} points")

    stats = {
        "registration_order": [i0, j0],
        "n_images": len(keypoints),
        "use_ba": use_ba,
        "use_geometric_verification": use_geometric_verification,
        "raw_match_pairs": raw_pairs,
        "raw_matches": raw_matches,
        "verified_match_pairs": len(match_graph),
        "verified_matches": sum(len(matches) for matches in match_graph.values()),
        "ba_history": [],
    }
    n_images = len(keypoints)
    since_ba = 0
    rejected: set[int] = set()  # failed this round; retried after map grows

    while len(registered) < n_images:
        # next image = most 2D-3D correspondences
        best_img, best_n = None, 0
        for img in range(n_images):
            if img in registered or img in rejected:
                continue
            p2, p3, _ = tm.correspondences_2d3d(img, keypoints, recon.points3d)
            if len(p2) > best_n:
                best_img, best_n = img, len(p2)
        # rejected is cleared on every success, so reaching here with all
        # remaining images rejected means no progress is possible
        if best_img is None or best_n < 15:
            log(f"stopping: no registrable image left ({best_n} correspondences)")
            break

        p2, p3, _ = tm.correspondences_2d3d(best_img, keypoints, recon.points3d)
        result = ransac_pnp(p3, p2, K)
        # accept on strong absolute support, decent ratio, or a small but
        # near-unanimous set (thin overlap at the edges of a capture);
        # contaminated tracks depress the ratio even when the pose is good
        ok = result is not None and (
            result[2].sum() >= 100
            or result[2].sum() >= max(30, 0.3 * len(p2))
            or (result[2].sum() >= 15 and result[2].sum() >= 0.6 * len(p2))
        )
        if not ok:
            n_inl = 0 if result is None else int(result[2].sum())
            log(f"image {best_img}: PnP rejected ({n_inl}/{len(p2)} inliers); deferring")
            rejected.add(best_img)
            continue
        R, t, mask = result
        recon.poses[best_img] = (R, t)
        registered.add(best_img)
        rejected.clear()  # map changed; rejected images may succeed now
        stats["registration_order"].append(best_img)
        n_new = _try_triangulate(tm, recon, keypoints, registered)
        log(
            f"registered image {best_img}: {mask.sum()}/{len(mask)} PnP inliers, "
            f"+{n_new} points (total {len(recon.points3d)})"
        )

        since_ba += 1
        if use_ba and since_ba >= ba_every:
            obs = _observations(tm, keypoints, registered)
            recon.poses, recon.points3d, rb, ra = run_bundle_adjustment(
                K, recon.poses, recon.points3d, obs
            )
            n_pruned = prune_outliers(tm, recon, keypoints, registered, max_err=4.0)
            log(f"  BA: rms {rb:.3f} -> {ra:.3f} px ({len(obs)} obs), pruned {n_pruned}")
            stats["ba_history"].append(
                {
                    "stage": "periodic",
                    "n_registered": len(registered),
                    "n_observations": len(obs),
                    "rms_before_px": rb,
                    "rms_after_px": ra,
                    "n_pruned": n_pruned,
                }
            )
            since_ba = 0

    # final BA + prune + BA
    if use_ba:
        obs = _observations(tm, keypoints, registered)
        recon.poses, recon.points3d, rb, ra = run_bundle_adjustment(
            K, recon.poses, recon.points3d, obs
        )
        log(f"final BA: rms {rb:.3f} -> {ra:.3f} px")
        stats["rms_before_final_ba"] = rb
        stats["ba_history"].append(
            {
                "stage": "final",
                "n_registered": len(registered),
                "n_observations": len(obs),
                "rms_before_px": rb,
                "rms_after_px": ra,
                "n_pruned": 0,
            }
        )

        n_pruned = prune_outliers(tm, recon, keypoints, registered)
        log(f"pruned {n_pruned} outlier points")
        obs = _observations(tm, keypoints, registered)
        recon.poses, recon.points3d, rb, ra = run_bundle_adjustment(
            K, recon.poses, recon.points3d, obs
        )
        log(f"post-prune BA: rms {rb:.3f} -> {ra:.3f} px")
        stats["rms_final"] = ra
        stats["ba_history"].append(
            {
                "stage": "post-prune",
                "n_registered": len(registered),
                "n_observations": len(obs),
                "rms_before_px": rb,
                "rms_after_px": ra,
                "n_pruned": n_pruned,
            }
        )

    obs = _observations(tm, keypoints, registered)
    if not use_ba:
        stats["rms_final"] = _reprojection_rms(recon, obs)
    stats["n_registered"] = len(registered)
    stats["n_points"] = len(recon.points3d)
    stats["n_observations"] = len(obs)
    stats["mean_track_length"] = len(obs) / max(len(recon.points3d), 1)
    return recon, tm, stats
