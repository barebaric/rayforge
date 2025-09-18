from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TraceConfig:
    """Parameters for a raster-to-vector tracing operation."""

    threshold: float = 0.5
