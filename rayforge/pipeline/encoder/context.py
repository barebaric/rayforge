from typing import Optional, List, Tuple, TYPE_CHECKING, Dict, Set
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
    """A container for variables available during G-code generation."""

    machine: "Machine"
    doc: "Doc"
    job: JobInfo
    # Assigning default values makes these fields optional in the constructor
    layer: Optional["Layer"] = None
    workpiece: Optional["WorkPiece"] = None

    @property
    def wcs_offset(self) -> Tuple[float, float, float]:
        """The (x, y, z) offset for the currently active WCS."""
        return self.machine.get_active_wcs_offset()

    # --- Static Variable Documentation ---
    _DOCS = {
        "job": [
            ("machine.name", _("The name of the current machine profile.")),
            (
                "machine.dimensions[0]",
                _("The width (X-axis) of the machine work area in mm."),
            ),
            (
                "machine.dimensions[1]",
                _("The height (Y-axis) of the machine work area in mm."),
            ),
            (
                "doc.name",
                _("The name of the current document file (if saved)."),
            ),
            (
                "job.extents[0]",
                _("The minimum X coordinate of the entire job."),
            ),
            (
                "job.extents[1]",
                _("The minimum Y coordinate of the entire job."),
            ),
            (
                "job.extents[2]",
                _("The maximum X coordinate of the entire job."),
            ),
            (
                "job.extents[3]",
                _("The maximum Y coordinate of the entire job."),
            ),
            ("wcs_offset[0]", _("The X offset of the currently active WCS.")),
            ("wcs_offset[1]", _("The Y offset of the currently active WCS.")),
            ("wcs_offset[2]", _("The Z offset of the currently active WCS.")),
        ],
        "layer": [
            (
                "layer.name",
                _("The name of the current layer being processed."),
            ),
        ],
        "workpiece": [
            (
                "workpiece.name",
                _("The name of the current workpiece being processed."),
            ),
            ("workpiece.pos[0]", _("The X position of the workpiece.")),
            ("workpiece.pos[1]", _("The Y position of the workpiece.")),
            ("workpiece.size[0]", _("The width of the workpiece.")),
            ("workpiece.size[1]", _("The height of the workpiece.")),
        ],
    }

    @classmethod
    def get_docs(cls, level: str) -> List[Tuple[str, str]]:
        """
        Gets all variables available up to a certain context level from the
        static documentation dictionary.
        """
        docs = cls._DOCS["job"]
        if level in ("layer", "workpiece"):
            docs = docs + cls._DOCS["layer"]
        if level == "workpiece":
            docs = docs + cls._DOCS["workpiece"]
        return sorted(docs, key=lambda item: item[0])

    @staticmethod
    def get_template_variable_docs() -> Dict[str, Set[str]]:
        """
        Returns a dictionary mapping G-code template keys to the set of
        variables they support. This is the single source of truth for
        dialect validation.
        """
        # Variables for movement commands
        move_vars = {"x", "y", "z", "f_command"}
        # Variables for cutting commands (inherits movement)
        cut_vars = move_vars.union({"i", "j", "s_command"})

        return {
            # Machine Control
            "laser_on": {"power"},
            "laser_off": set(),
            "tool_change": {"tool_number"},
            "set_speed": {"speed"},
            "air_assist_on": set(),
            "air_assist_off": set(),
            "home_all": set(),
            "home_axis": {"axis_letter"},
            "move_to": {"speed", "x", "y", "z"},
            "jog": {"speed"},
            "clear_alarm": set(),
            "set_wcs_offset": {"p_num", "x", "y", "z"},
            "probe_cycle": {"axis_letter", "max_travel", "feed_rate"},
            # Movement
            "travel_move": move_vars,
            "linear_move": cut_vars,
            "arc_cw": cut_vars,
            "arc_ccw": cut_vars,
        }
