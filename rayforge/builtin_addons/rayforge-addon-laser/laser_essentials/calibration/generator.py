"""
The base class of all calibration test generators.

A generator is pure: from its parameters and the machine it computes a
:class:`CalibrationLayout`, without touching the document or the UI.
``CalibrationBuilder`` turns that layout into document content and
``CalibrationCmd`` keeps it up to date when the parameters change.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from rayforge.core.varset import VarSet
from rayforge.pipeline.transformer.registry import transformer_registry

from ..steps import ContourStep, EngraveStep
from ..steps.laser_step import LaserStep
from .layout import CalibrationLayout

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


#: Item labels and item subtitles of a two-row tuple setting.
RowLabels = tuple[tuple[str, str], tuple[str, str]]


class CalibrationGenerator(ABC):
    """Describes one kind of calibration test."""

    id: ClassVar[str]

    @property
    @abstractmethod
    def title(self) -> str:
        """The name of the test, also used for its layer."""

    @abstractmethod
    def varset(self, context: "RayforgeContext") -> VarSet:
        """The settings of the test, with their defaults."""

    @abstractmethod
    def layout(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> CalibrationLayout:
        """
        The test for *params*. Raises ``ValueError`` with a message for
        the user when the parameters do not make a valid test.
        """

    def defaults(self, context: "RayforgeContext") -> dict[str, Any]:
        """The parameters of a new test."""
        return self.varset(context).get_values()

    def is_available(self, context: "RayforgeContext") -> bool:
        """Whether the current machine can run this test."""
        return context.machine is not None

    def apply_change(
        self,
        params: dict[str, Any],
        keys: set[str],
        context: "RayforgeContext",
    ) -> dict[str, Any]:
        """
        Follow-up changes after the user edited *keys*, such as filling
        in a preset. Returns the parameters to use.
        """
        return params

    def row_labels(self, params: dict[str, Any]) -> dict[str, RowLabels]:
        """Item labels and subtitles of tuple rows that depend on params."""
        return {}

    def help_text(self, params: dict[str, Any]) -> str:
        """Instructions shown with the settings."""
        return ""


def contour_step(
    context: "RayforgeContext", power_percent: float, speed: float
) -> ContourStep:
    step = ContourStep.create(context)
    set_laser(step, power_percent, speed)
    return step


def engrave_step(
    context: "RayforgeContext", power_percent: float, speed: float
) -> EngraveStep:
    step = EngraveStep.create(context)
    set_laser(step, power_percent, speed)
    _update_overscan(context, step)
    return step


def set_laser(step: LaserStep, power_percent: float, speed: float):
    step.set_power(power_percent / 100.0)
    step.set_cut_speed(int(speed))


def _update_overscan(context: "RayforgeContext", step: LaserStep):
    """Recomputes the automatic overscan for the step's speed."""
    machine = context.machine
    overscan = cast(
        "OverscanTransformerType | None",
        transformer_registry.get("OverscanTransformer"),
    )
    if machine is None or overscan is None:
        return
    distance = overscan.calculate_auto_distance(
        step.cut_speed, machine.acceleration
    )
    for spec in step.per_workpiece_transformers_dicts:
        if spec.get("name") == "OverscanTransformer" and spec.get("auto"):
            spec["distance_mm"] = distance
