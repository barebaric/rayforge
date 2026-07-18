from typing import TYPE_CHECKING

from raygeo.ops import Ops

from ...machine.models.dialect import GcodeDialect
from .base import EncodedOutput, MachineCodeOpMap, OpsEncoder
from .rust_helpers import build_encode_context, dialect_to_spec

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine


class GcodeEncoder(OpsEncoder):
    """Converts Ops commands to G-code via the Rust encoder in raygeo."""

    def __init__(self, dialect: GcodeDialect):
        self.dialect: GcodeDialect = dialect

    @classmethod
    def for_machine(cls, machine: "Machine") -> "GcodeEncoder":
        assert machine.dialect is not None
        return cls(machine.dialect)

    def encode(
        self, ops: Ops, machine: "Machine", doc: "Doc"
    ) -> EncodedOutput:
        dialect_spec = dialect_to_spec(self.dialect, machine)
        context = build_encode_context(ops, machine, doc)
        result = ops.to_gcode(dialect_spec, context)
        return EncodedOutput(
            text=result["text"],
            op_map=MachineCodeOpMap(
                op_to_machine_code=result["op_to_machine_code"],
                machine_code_to_op=result["machine_code_to_op"],
            ),
        )
