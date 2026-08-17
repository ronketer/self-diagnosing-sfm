"""Graph connectivity, failure stats, and diagnostic helpers for SfM."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ..models import Reconstruction
from ..tracks import TrackManager
from ..validation.diagnostics import ReconstructionDiagnostics


def compute_view_graph_components(
    num_images: int,
    verified_pairs: list[tuple[int, int]],
    registered_images: set[int],
) -> dict[str, Any]:
    """Compute connected components of the verified two-view graph."""
    adjacency: dict[int, set[int]] = {
        i: set()
        for i in range(num_images)
    }

    for image_a, image_b in verified_pairs:
        adjacency[image_a].add(image_b)
        adjacency[image_b].add(image_a)

    visited: set[int] = set()
    components: list[list[int]] = []

    for image_id in range(num_images):
        if image_id in visited:
            continue

        component: list[int] = []
        queue = deque([image_id])
        visited.add(image_id)

        while queue:
            current = queue.popleft()
            component.append(current)

            for neighbor in sorted(
                adjacency[current]
            ):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        components.append(
            sorted(component)
        )

    components.sort(
        key=lambda component: (
            -len(component),
            component,
        )
    )

    active_component: list[int] = []
    other_components: list[list[int]] = []
    isolated_images: list[int] = []

    for component in components:
        if (
            len(component) == 1
            and not adjacency[component[0]]
        ):
            isolated_images.append(
                component[0]
            )

        elif any(
            image_id in registered_images
            for image_id in component
        ):
            active_component.extend(
                component
            )

        else:
            other_components.append(
                component
            )

    return {
        "num_components": len(components),
        "components": components,
        "active_component": sorted(
            set(active_component)
        ),
        "other_components": other_components,
        "isolated_images": sorted(
            isolated_images
        ),
    }


def _failure_stats(
    *,
    reconstruction: Reconstruction,
    track_manager: TrackManager,
    diagnostics: ReconstructionDiagnostics,
    n_images: int,
    raw_pairs: int,
    raw_matches: int,
    verified_pairs: int,
    verified_matches: int,
    use_ba: bool,
    use_geometric_verification: bool,
) -> dict[str, Any]:
    """Build stats for a reconstruction that failed before initialization."""
    diagnostics.summary_stats = {
        "initial_pair": [],
        "registered_images": 0,
        "total_images": n_images,
        "landmarks": 0,
        "observations": 0,
        "mean_track_length": 0.0,
        "final_reprojection_rms_px": None,
        "status": diagnostics.status.value,
        "termination_reason": (
            diagnostics
            .termination_reason
            .value
        ),
        "registration_order": [],
    }

    return {
        "registration_order": [],
        "n_registered": 0,
        "n_images": n_images,
        "n_points": 0,
        "n_observations": 0,
        "mean_track_length": 0.0,
        "rms_before_final_ba": None,
        "rms_final": None,
        "use_ba": use_ba,
        "use_geometric_verification": (
            use_geometric_verification
        ),
        "raw_match_pairs": raw_pairs,
        "raw_matches": raw_matches,
        "verified_match_pairs": (
            verified_pairs
        ),
        "verified_matches": (
            verified_matches
        ),
        "ba_history": [],
        "diagnostics": diagnostics,
        "status": (
            diagnostics.status.value
        ),
        "termination_reason": (
            diagnostics
            .termination_reason
            .value
        ),
        "initial_pair_verified_matches": (
            np.zeros(
                (0, 2),
                dtype=np.int64,
            )
        ),
    }
