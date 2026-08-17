"""Geometry package: projection, two-view epipolar geometry, and poses."""

from .essential import (
    choose_pose,
    decompose_essential,
    estimate_essential_8pt,
    hartley_normalize,
    ransac_essential,
    sampson_distance,
)
from .pose import (
    relative_pose,
    rotation_error_deg,
    umeyama,
)
from .projection import (
    camera_center,
    point_depths,
    reproject,
    to_normalized,
    triangulation_angles_deg,
)

__all__ = [
    "camera_center",
    "choose_pose",
    "decompose_essential",
    "estimate_essential_8pt",
    "hartley_normalize",
    "point_depths",
    "ransac_essential",
    "relative_pose",
    "reproject",
    "rotation_error_deg",
    "sampson_distance",
    "to_normalized",
    "triangulation_angles_deg",
    "umeyama",
]
