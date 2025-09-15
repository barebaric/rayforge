from typing import Optional, List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...core.layer import Layer
    from ...core.workpiece import WorkPiece


@dataclass
class JobInfo:
    """Information about the entire job."""

    extents: Tuple[float, float, float, float]


@dataclass
class GcodeContext:
    """A container for variables available during G--code generation."""

    machine: "Machine"
    doc: "Doc"
    job: JobInfo
    # Assigning default values makes these fields optional in the constructor
    layer: Optional["Layer"] = None
    workpiece: Optional["WorkPiece"] = None

    # --- Static Variable Documentation ---
    _DOCS = {
        "job": [
            ("machine.name", "The name of the current machine profile."),
            (
                "machine.dimensions[0]",
                "The width (X-axis) of the machine work area in mm.",
            ),
            (
                "machine.dimensions[1]",
                "The height (Y-axis) of the machine work area in mm.",
            ),
            ("doc.name", "The name of the current document file (if saved)."),
            ("job.extents[0]", "The minimum X coordinate of the entire job."),
            ("job.extents[1]", "The minimum Y coordinate of the entire job."),
            ("job.extents[2]", "The maximum X coordinate of the entire job."),
            ("job.extents[3]", "The maximum Y coordinate of the entire job."),
        ],
        "layer": [
            ("layer.name", "The name of the current layer being processed."),
        ],
        "workpiece": [
            (
                "workpiece.name",
                "The name of the current workpiece being processed.",
            ),
            ("workpiece.pos[0]", "The X position of the workpiece."),
            ("workpiece.pos[1]", "The Y position of the workpiece."),
            ("workpiece.size[0]", "The width of the workpiece."),
            ("workpiece.size[1]", "The height of the workpiece."),
        ],
    }

    @classmethod
    def get_docs(cls, level: str) -> List[Tuple[str, str]]:
        """
        Gets all variables available up to a certain context level from the
        static documentation dictionary.
        """
        if level == "workpiece_hook":
            return (
                cls._DOCS["job"] + cls._DOCS["layer"] + cls._DOCS["workpiece"]
            )
        if level == "layer_hook":
            return cls._DOCS["job"] + cls._DOCS["layer"]
        # Default for job hooks and macros
        return cls._DOCS["job"]
