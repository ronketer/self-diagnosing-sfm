"""Validation and diagnostics package."""

from .degeneracy import (
    check_cheirality,
    check_parallax,
    check_planar_degeneracy,
    check_pure_rotation,
)
from .diagnostics import (
    DiagnosticEvent,
    FailureReason,
    ReconstructionDiagnostics,
)

__all__ = [
    "DiagnosticEvent",
    "FailureReason",
    "ReconstructionDiagnostics",
    "check_cheirality",
    "check_parallax",
    "check_planar_degeneracy",
    "check_pure_rotation",
]
