"""Validation and diagnostics package."""

from .degeneracy import (
    check_cheirality,
    check_ground_truth_baseline,
    check_parallax,
    check_planar_degeneracy,
    check_pure_rotation,
    evaluate_homography_support,
)
from .diagnostics import (
    CameraRegistrationStatus,
    DiagnosticEvent,
    FailureReason,
    ReconstructionDiagnostics,
    ReconstructionStatus,
    TerminationReason,
)

__all__ = [
    "CameraRegistrationStatus",
    "DiagnosticEvent",
    "FailureReason",
    "ReconstructionDiagnostics",
    "ReconstructionStatus",
    "TerminationReason",
    "check_cheirality",
    "check_ground_truth_baseline",
    "check_parallax",
    "check_planar_degeneracy",
    "check_pure_rotation",
    "evaluate_homography_support",
]
