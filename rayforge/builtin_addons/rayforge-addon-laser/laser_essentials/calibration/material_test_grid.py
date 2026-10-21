"""
The Material Test Grid: one procedural workpiece with one Material Test
step that runs every cell of the grid.

The cells, labels and their risk-optimized order are computed by the
raygeo ``material_test_grid`` assembler; this generator only decides
the parameters and the size of the workpiece.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.core.step import Step
from rayforge.core.varset import LabeledChoiceVar, TupleVar, VarSet

from ..material_test_helpers import (
    GridMode,
    draw_material_test_preview,
    get_material_test_proportional_size,
)
from ..steps import MaterialTestStep
from .generator import CalibrationGenerator, RowLabels
from .layout import CalibrationLayout, CellEntry, ProceduralShape

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext

#: The step attributes that make up the test; the laser settings of the
#: step stay in its own settings dialog.
PARAM_KEYS = (
    "test_type",
    "grid_mode",
    "speed_range",
    "power_range",
    "passes_range",
    "offset_range",
    "fixed_speed",
    "fixed_power",
    "grid_dimensions",
    "shape_size",
    "spacing",
    "include_labels",
    "label_power_percent",
    "label_speed",
    "line_interval_mm",
)

NO_PRESET = "none"

PRESETS: dict[str, dict[str, Any]] = {
    "Diode Engrave": {
        "test_type": "Engrave",
        "speed_range": (1000.0, 10000.0),
        "power_range": (10.0, 100.0),
    },
    "Diode Cut": {
        "test_type": "Cut",
        "speed_range": (100.0, 5000.0),
        "power_range": (50.0, 100.0),
    },
    "CO2 Engrave": {
        "test_type": "Engrave",
        "speed_range": (3000.0, 20000.0),
        "power_range": (10.0, 50.0),
    },
    "CO2 Cut": {
        "test_type": "Cut",
        "speed_range": (1000.0, 20000.0),
        "power_range": (30.0, 100.0),
    },
}

#: Line interval that makes row-to-row misalignment of a bidirectional
#: scan visible by eye in Speed vs Offset mode.
OFFSET_TEST_LINE_INTERVAL = 0.5


#: Rows of the settings in the order of the old settings page: grid,
#: labels, then the ranges the grid varies.
_ROW_ORDER = (
    "test_type",
    "grid_mode",
    "grid_dimensions",
    "shape_size",
    "spacing",
    "line_interval_mm",
    "include_labels",
    "label_power_percent",
    "label_speed",
    "fixed_speed",
    "fixed_power",
    "power_range",
    "speed_range",
    "passes_range",
    "offset_range",
)


def _preset_labels() -> dict[str, str]:
    return {
        "Diode Engrave": _("Diode Engrave"),
        "Diode Cut": _("Diode Cut"),
        "CO2 Engrave": _("CO2 Engrave"),
        "CO2 Cut": _("CO2 Cut"),
    }


def _dimension_labels() -> dict[str, RowLabels]:
    return {
        GridMode.POWER_VS_SPEED.value: (
            (_("Columns (Power Steps)"), _("Rows (Speed Steps)")),
            (_("Number of power variations"), _("Number of speed variations")),
        ),
        GridMode.POWER_VS_PASSES.value: (
            (_("Columns (Power Steps)"), _("Rows (Passes Steps)")),
            (
                _("Number of power variations"),
                _("Number of passes variations"),
            ),
        ),
        GridMode.SPEED_VS_PASSES.value: (
            (_("Columns (Speed Steps)"), _("Rows (Passes Steps)")),
            (
                _("Number of speed variations"),
                _("Number of passes variations"),
            ),
        ),
        GridMode.SPEED_VS_OFFSET.value: (
            (_("Columns (Speed Steps)"), _("Rows (Offset Steps)")),
            (
                _("Number of speed variations"),
                _("Number of offset variations"),
            ),
        ),
    }


def params_from_step(step: Step) -> dict[str, Any]:
    """The test parameters stored on a Material Test step."""
    params = {key: getattr(step, key, None) for key in PARAM_KEYS}
    if params["line_interval_mm"] is None:
        params["line_interval_mm"] = 0.0
    return params


def _max_speed(context: "RayforgeContext") -> float | None:
    machine = context.machine
    return float(machine.max_cut_speed) if machine is not None else None


class MaterialTestGridGenerator(CalibrationGenerator):
    id = "material_test_grid"

    @property
    def title(self) -> str:
        return _("Material Test Grid")

    def varset(self, context: "RayforgeContext") -> VarSet:
        step_vars = {var.key: var for var in MaterialTestStep.recipe_varset()}
        speed_range = step_vars["speed_range"]
        max_speed = _max_speed(context)
        if isinstance(speed_range, TupleVar) and max_speed is not None:
            speed_range.max_val = max_speed
        labels = _preset_labels()
        preset = LabeledChoiceVar(
            key="preset",
            label=_("Presets"),
            description=_("Load common test configurations"),
            choices=[(_("Select"), NO_PRESET)]
            + [(labels[key], key) for key in PRESETS],
            default=NO_PRESET,
            allow_none=False,
        )
        return VarSet(vars=[preset] + [step_vars[k] for k in _ROW_ORDER])

    def defaults(self, context: "RayforgeContext") -> dict[str, Any]:
        params = super().defaults(context)
        max_speed = _max_speed(context)
        if max_speed is not None:
            low, high = params["speed_range"]
            params["speed_range"] = (min(low, max_speed), min(high, max_speed))
        return params

    def apply_change(
        self,
        params: dict[str, Any],
        keys: set[str],
        context: "RayforgeContext",
    ) -> dict[str, Any]:
        result = dict(params)
        if "preset" in keys and result.get("preset") in PRESETS:
            result.update(PRESETS[result["preset"]])
            max_speed = _max_speed(context)
            if max_speed is not None:
                low, high = result["speed_range"]
                result["speed_range"] = (
                    min(low, max_speed),
                    min(high, max_speed),
                )
        if (
            "grid_mode" in keys
            and result.get("grid_mode") == GridMode.SPEED_VS_OFFSET.value
        ):
            result["test_type"] = "Engrave"
            result["line_interval_mm"] = OFFSET_TEST_LINE_INTERVAL
        return result

    def row_labels(self, params: dict[str, Any]) -> dict[str, RowLabels]:
        labels = _dimension_labels()
        mode = params.get("grid_mode", GridMode.POWER_VS_SPEED.value)
        return {
            "grid_dimensions": labels.get(
                mode, labels[GridMode.POWER_VS_SPEED.value]
            )
        }

    def help_text(self, params: dict[str, Any]) -> str:
        return _(
            "The laser head and the post-processing of the grid stay in "
            "the settings of its operation."
        )

    def layout(
        self, params: dict[str, Any], context: "RayforgeContext"
    ) -> CalibrationLayout:
        grid_params = self._grid_params(params)
        step = MaterialTestStep.create(context)
        step.name = self.title
        for key, value in grid_params.items():
            setattr(step, key, value)
        if not step.line_interval_mm:
            step.line_interval_mm = None
        shape = ProceduralShape(
            draw_function=draw_material_test_preview,
            size_function=get_material_test_proportional_size,
            params=grid_params,
        )
        return CalibrationLayout([CellEntry(self.title, shape, step)])

    @staticmethod
    def _grid_params(params: dict[str, Any]) -> dict[str, Any]:
        """
        The step attributes for *params*. Counts are whole numbers even
        when a spin row or a saved project gives them as floats.
        """
        result = {key: params[key] for key in PARAM_KEYS}
        for key in ("speed_range", "power_range", "offset_range"):
            result[key] = tuple(float(v) for v in result[key])
        for key in ("passes_range", "grid_dimensions"):
            result[key] = tuple(round(v) for v in result[key])
        cols, rows = result["grid_dimensions"]
        if min(cols, rows) < 2:
            raise ValueError(
                _("A Material Test Grid needs at least 2×2 cells.")
            )
        return result
