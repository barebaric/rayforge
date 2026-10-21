"""
The base class of all calibration test generators.

A generator is pure: from its parameters and the machine it computes a
:class:`CalibrationLayout`, without touching the document or the UI.
``CalibrationBuilder`` turns that layout into document content and
``CalibrationCmd`` keeps it up to date when the parameters change.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from rayforge.core.step import Step
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

    #: Whether a rebuild keeps the existing steps and only changes the
    #: settings in :meth:`step_settings`, so what the user set in a
    #: step's own settings (laser head, post-processing) stays.
    keeps_steps: ClassVar[bool] = False

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

    def settings_varset(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> VarSet:
        """The settings shown for an existing test with *params*."""
        return self.varset(context)

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

    def warning(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> str:
        """Why the test cannot run as it is on the current machine."""
        return ""

    def step_settings(self, step: Step) -> dict[str, Any]:
        """
        The step attributes this generator controls, for a rebuild that
        keeps the existing steps (see :attr:`keeps_steps`).
        """
        return {}

    def params_from_steps(
        self, params: dict[str, Any], steps: list[Step]
    ) -> dict[str, Any]:
        """
        The parameters of a test whose steps may have changed outside
        the Test Settings, for example by applying a recipe.
        """
        return params


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
