from __future__ import annotations

from typing import Callable, List

from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry

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


def create_contour_step(context, name=None, optimize: bool = True) -> Step:
    return ContourStep.create(context, name=name, optimize=optimize)


def create_engrave_step(context, name=None) -> Step:
    return EngraveStep.create(context, name=name)


def create_shrinkwrap_step(context, name=None) -> Step:
    return ShrinkWrapStep.create(context, name=name)


def create_frame_step(context, name=None) -> Step:
    return FrameStep.create(context, name=name)


def create_material_test_step(context, name=None) -> Step:
    return MaterialTestStep.create(context, name=name)


STEP_FACTORIES: List[Callable] = [
    ContourStep.create,
    EngraveStep.create,
    ShrinkWrapStep.create,
    FrameStep.create,
]

__all__ = [
    "ContourStep",
    "EngraveStep",
    "FrameStep",
    "MaterialTestStep",
    "ShrinkWrapStep",
    "create_contour_step",
    "create_engrave_step",
    "create_shrinkwrap_step",
    "create_frame_step",
    "create_material_test_step",
    "STEP_FACTORIES",
]
