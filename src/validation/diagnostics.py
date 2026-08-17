"""Diagnostic models and logging for Structure-from-Motion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class FailureReason(str, Enum):
    INSUFFICIENT_MATCHES = "INSUFFICIENT_MATCHES"
    ESSENTIAL_ESTIMATION_FAILED = "ESSENTIAL_ESTIMATION_FAILED"
    LOW_PARALLAX = "LOW_PARALLAX"
    POOR_CHEIRALITY = "POOR_CHEIRALITY"
    HIGH_REPROJECTION_ERROR = "HIGH_REPROJECTION_ERROR"
    INSUFFICIENT_PNP_CORRESPONDENCES = "INSUFFICIENT_PNP_CORRESPONDENCES"
    PNP_FAILED = "PNP_FAILED"
    INSUFFICIENT_PNP_INLIERS = "INSUFFICIENT_PNP_INLIERS"
    NEGATIVE_DEPTH = "NEGATIVE_DEPTH"
    TRACK_CONFLICT = "TRACK_CONFLICT"


@dataclass
class DiagnosticEvent:
    """An event recording a validation or decision point in the reconstruction pipeline."""
    stage: str
    entity: str
    status: str
    reason: FailureReason | str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "stage": self.stage,
            "entity": self.entity,
            "status": self.status,
            "reason": self.reason.value if isinstance(self.reason, FailureReason) else self.reason,
            "metrics": self.metrics,
        }
        return d


@dataclass
class ReconstructionDiagnostics:
    """Collector of diagnostic events, pair evaluations, and reconstruction quality stats."""
    events: list[DiagnosticEvent] = field(default_factory=list)
    initial_pair_evaluations: list[dict[str, Any]] = field(default_factory=list)
    registration_events: list[dict[str, Any]] = field(default_factory=list)
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def record(self, event: DiagnosticEvent) -> None:
        self.events.append(event)
        if event.stage == "CAMERA_REGISTRATION":
            self.registration_events.append(event.to_dict())

    def record_pair_evaluation(self, eval_dict: dict[str, Any]) -> None:
        self.initial_pair_evaluations.append(eval_dict)

    def get_rejections(self) -> list[dict[str, Any]]:
        return [
            e.to_dict()
            for e in self.events
            if e.status in ("REJECTED", "DEFERRED", "FAILED")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary_stats,
            "rejections": self.get_rejections(),
            "initial_pair_evaluations": self.initial_pair_evaluations,
            "events": [e.to_dict() for e in self.events],
        }
