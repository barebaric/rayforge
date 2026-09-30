"""The Command step: inject raw machine code at a workflow position.

Phase 1 of the "Command step" feature (issue #449). The step holds a
multi-line text block; each non-empty line is emitted as one raygeo
``Custom`` command at the step's exact position in the layer's
workflow. The text is stored unexpanded in the project and reaches
the encoder verbatim aside from path-variable expansion
(``machine.*``, ``layer.*``, ``job.*``, ``wcs_offset[*]``), which the
G-code encoder performs — ``layer.*`` variables resolve correctly for
lines emitted mid-layer.
"""

from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.cnc.execution.specs import ComputePayload
from raygeo.ops.assembly import Assembler
from raygeo.ops.assembly.custom import CustomSpec
from raygeo.ops.part import Part

from rayforge.core.step import Step, legacy_producer_params
from rayforge.core.varset import CodeVar, VarSet
from rayforge.pipeline.intent_builder import _driver_uses_gcode

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.machine.models.machine import Machine


class CommandStep(Step):
    """Inject user-provided machine-code lines into the job."""

    TYPELABEL = _("Command")
    ICON = "step-command-symbolic"
    ASSEMBLER_NAME = "custom"
    needs_workpieces = False

    def __init__(
        self,
        name: str | None = None,
        typelabel: str | None = None,
    ):
        super().__init__(typelabel=typelabel or self.TYPELABEL, name=name)
        self.command_text: str = ""

    # -- Configuration -----------------------------------------------------

    @classmethod
    def create(
        cls,
        context: RayforgeContext,
        name: str | None = None,
        **kwargs,
    ) -> CommandStep:
        machine = context.machine
        step = cls(name=name)
        if machine is not None:
            step.max_cut_speed = machine.max_cut_speed
            step.max_travel_speed = machine.max_travel_speed
        return step

    @property
    def command_lines(self) -> list[str]:
        """The non-empty lines of the command text, in order."""
        return [
            line for line in self.command_text.splitlines() if line.strip()
        ]

    def set_command_text(self, text: str) -> None:
        """Sets the machine-code text and notifies listeners."""
        if self.command_text != text:
            self.command_text = text
            self.updated.send(self)

    # -- Pipeline ----------------------------------------------------------

    def build_command_payload(
        self,
        machine: Machine,
    ) -> tuple[Part, ComputePayload]:
        """One compute node for the whole step: the raw text lines.

        The step is geometry-less — the part carries no geometry and
        the assembler emits one ``Custom`` command per line, so the
        node's output is independent of the layer's workpieces.
        """
        part = Part(size_mm=(0.0, 0.0))
        payload = ComputePayload(
            assembler=Assembler(CustomSpec(lines=self.command_lines))
        )
        return part, payload

    def get_cache_params(self) -> dict[str, Any]:
        params = super().get_cache_params()
        params["command_text"] = self.command_text
        return params

    # -- Recipes -----------------------------------------------------------

    @classmethod
    def recipe_varset(cls) -> VarSet:
        return VarSet(
            vars=[
                CodeVar(
                    key="command_text",
                    label=_("Machine Code"),
                    default="",
                )
            ]
        )

    # -- UI helpers --------------------------------------------------------

    @property
    def show_general_settings(self) -> bool:
        return False

    def get_summary(self) -> str:
        lines = self.command_lines
        if not lines:
            return _("No machine code")
        if len(lines) == 1:
            return lines[0]
        return _("{count} lines").format(count=len(lines))

    def check(self, machine) -> list[str]:
        warnings: list[str] = []
        if not self.command_lines:
            warnings.append(_("The command step has no machine code."))
        if machine is None:
            return warnings
        if not _driver_uses_gcode(machine):
            warnings.append(
                _(
                    "The '{name}' driver does not support custom "
                    "machine code; the lines are not emitted for "
                    "this machine."
                ).format(name=machine.driver_name or _("None"))
            )
        return warnings

    # -- Serialization -----------------------------------------------------

    def to_dict(self) -> dict:
        result = super().to_dict()
        result["command_text"] = self.command_text
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        step = super().from_dict(data)
        assert isinstance(step, cls)
        params = legacy_producer_params(data)
        step.command_text = data.get(
            "command_text", params.get("command_text", "")
        )
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset({"command_text"})
