"""Diagnostic models, status enums, and structured quality logging for Structure-from-Motion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReconstructionStatus(str, Enum):
    """Overall reconstruction outcome status."""
    SUCCESS = "SUCCESS"    # All images registered
    PARTIAL = "PARTIAL"    # Valid reconstruction exists, but some images remain unregistered
    FAILED = "FAILED"      # No valid initialization or no usable reconstruction


class TerminationReason(str, Enum):
    """Explicit reason explaining why the reconstruction halted."""
    COMPLETED = "COMPLETED"                                  # All cameras registered successfully
    NO_2D3D_BRIDGE = "NO_2D3D_BRIDGE"                        # Verified match graph is disconnected; remaining cameras have 0 correspondences
    NO_REGISTRABLE_CAMERA = "NO_REGISTRABLE_CAMERA"          # Remaining cameras have insufficient correspondences or failed PnP
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"          # No candidate pair met geometric criteria
    INSUFFICIENT_VERIFIED_GEOMETRY = "INSUFFICIENT_VERIFIED_GEOMETRY"


class CameraRegistrationStatus(str, Enum):
    """Final diagnostic status for each individual input image."""
    REGISTERED = "REGISTERED"
    BLOCKED_NO_2D3D_CORRESPONDENCES = "BLOCKED_NO_2D3D_CORRESPONDENCES"
    REJECTED_PNP = "REJECTED_PNP"
    REJECTED_GEOMETRY = "REJECTED_GEOMETRY"
    NOT_REACHED = "NOT_REACHED"


class FailureReason(str, Enum):
    """Controlled categorization of geometric and numerical failure modes."""
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    ESSENTIAL_ESTIMATION_FAILED = "ESSENTIAL_ESTIMATION_FAILED"
    LOW_PARALLAX = "LOW_PARALLAX"
    POOR_CHEIRALITY = "POOR_CHEIRALITY"
    HIGH_REPROJECTION_ERROR = "HIGH_REPROJECTION_ERROR"
    INSUFFICIENT_PNP_CORRESPONDENCES = "INSUFFICIENT_PNP_CORRESPONDENCES"
    PNP_FAILED = "PNP_FAILED"
    INSUFFICIENT_PNP_INLIERS = "INSUFFICIENT_PNP_INLIERS"
    LOW_PNP_INLIER_RATIO = "LOW_PNP_INLIER_RATIO"
    HIGH_PNP_REPROJECTION_ERROR = "HIGH_PNP_REPROJECTION_ERROR"
    NEGATIVE_DEPTH = "NEGATIVE_DEPTH"
    TRACK_CONFLICT = "TRACK_CONFLICT"
    HOMOGRAPHY_DOMINANT = "HOMOGRAPHY_DOMINANT"
    NONFINITE_TRIANGULATION = "NONFINITE_TRIANGULATION"


@dataclass
class DiagnosticEvent:
    """An event recording a validation or decision point in the reconstruction pipeline."""
    stage: str
    entity: str
    status: str
    reason: FailureReason | str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "entity": self.entity,
            "status": self.status,
            "reason": self.reason.value if isinstance(self.reason, FailureReason) else self.reason,
            "metrics": self.metrics,
        }


@dataclass
class ReconstructionDiagnostics:
    """Collector of diagnostic events, pair evaluations, camera statuses, and reconstruction quality stats."""
    status: ReconstructionStatus = ReconstructionStatus.PARTIAL
    termination_reason: TerminationReason | None = None

    view_graph: dict[str, Any] = field(
        default_factory=lambda: {
            "num_components": 0,
            "active_component": [],
            "other_components": [],
            "isolated_images": [],
        }
    )

    camera_status: dict[str, dict[str, Any]] = field(default_factory=dict)
    triangulation_rejections: dict[str, int] = field(default_factory=dict)
    pnp_rejections: dict[str, int] = field(default_factory=dict)
    track_conflict_rejected_edges: int = 0
    landmark_removals: int = 0

    events: list[DiagnosticEvent] = field(default_factory=list)
    initial_pair_evaluations: list[dict[str, Any]] = field(default_factory=list)
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def record(self, event: DiagnosticEvent) -> None:
        self.events.append(event)

    def record_triangulation_rejection(self, reason: FailureReason | str) -> None:
        key = reason.value if isinstance(reason, FailureReason) else str(reason)
        self.triangulation_rejections[key] = self.triangulation_rejections.get(key, 0) + 1

    def record_pnp_rejection(self, reason: FailureReason | str) -> None:
        key = reason.value if isinstance(reason, FailureReason) else str(reason)
        self.pnp_rejections[key] = self.pnp_rejections.get(key, 0) + 1

    def record_pair_evaluation(self, eval_dict: dict[str, Any]) -> None:
        self.initial_pair_evaluations.append(eval_dict)

    def get_rejections(self) -> list[dict[str, Any]]:
        return [
            e.to_dict()
            for e in self.events
            if e.status in ("REJECTED", "DEFERRED", "FAILED", "BLOCKED")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value if isinstance(self.status, ReconstructionStatus) else self.status,
            "termination_reason": (
                self.termination_reason.value
                if isinstance(self.termination_reason, TerminationReason)
                else self.termination_reason
            ),
            "registered_images": int(self.summary_stats.get("registered_images", 0)),
            "total_images": int(self.summary_stats.get("total_images", 0)),
            "registered_image_ids": self.summary_stats.get("registration_order", []),
            "landmarks": int(self.summary_stats.get("landmarks", 0)),
            "observations": int(self.summary_stats.get("observations", 0)),
            "final_reprojection_rms_px": self.summary_stats.get("final_reprojection_rms_px"),
            "view_graph": self.view_graph,
            "camera_status": self.camera_status,
            "triangulation_rejections": self.triangulation_rejections,
            "pnp_rejections": self.pnp_rejections,
            "track_conflict_rejected_edges": self.track_conflict_rejected_edges,
            "landmark_removals": self.landmark_removals,
            "summary": self.summary_stats,
            "rejections": self.get_rejections(),
            "initial_pair_evaluations": self.initial_pair_evaluations,
            "events": [e.to_dict() for e in self.events],
        }
