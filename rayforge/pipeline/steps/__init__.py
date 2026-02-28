from __future__ import annotations

from ...core.step_registry import step_registry
from .contour import ContourStep
from .engrave import EngraveStep
from .frame import FrameStep
from .material_test import MaterialTestStep
from .shrinkwrap import ShrinkWrapStep

step_registry.register(ContourStep)
step_registry.register(EngraveStep)
step_registry.register(FrameStep)
step_registry.register(MaterialTestStep)
step_registry.register(ShrinkWrapStep)

__all__ = [
    "ContourStep",
    "EngraveStep",
    "FrameStep",
    "MaterialTestStep",
    "ShrinkWrapStep",
]
