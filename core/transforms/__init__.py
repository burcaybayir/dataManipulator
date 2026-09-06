"""The transform library — every distortion technique the app can demonstrate.

Importing this package registers all built-in transforms. To add a technique,
write a module here and decorate its class with
:func:`~core.transforms.registry.register`; nothing else needs to change.
"""

from __future__ import annotations

from core.transforms.aggregation_swap import AggregationSwap
from core.transforms.aspect_ratio import AspectRatioChange
from core.transforms.axis_inversion import InvertedAxis
from core.transforms.axis_truncation import TruncatedAxis
from core.transforms.base import Transform, TransformError
from core.transforms.bin_regrouping import BinRegrouping
from core.transforms.cumulative import Cumulative
from core.transforms.dual_axis import DualAxis
from core.transforms.outlier_drop import OutlierDrop
from core.transforms.ratio_vs_absolute import RatioVsAbsolute
from core.transforms.rebase_index import RebaseIndex
from core.transforms.registry import available, build, from_dict, get, register
from core.transforms.smoothing import Smoothing
from core.transforms.stack import MAX_DEPTH, TransformStack
from core.transforms.window import CherryPickedWindow

__all__ = [
    "MAX_DEPTH",
    "AggregationSwap",
    "AspectRatioChange",
    "BinRegrouping",
    "CherryPickedWindow",
    "Cumulative",
    "DualAxis",
    "InvertedAxis",
    "OutlierDrop",
    "RatioVsAbsolute",
    "RebaseIndex",
    "Smoothing",
    "Transform",
    "TransformError",
    "TransformStack",
    "TruncatedAxis",
    "available",
    "build",
    "from_dict",
    "get",
    "register",
]
