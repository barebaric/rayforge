"""
Calibration tests: the Material Test Grid, the Interval Test and the
Focus Test share one generator architecture.

Each test is a :class:`CalibrationGenerator` that turns its parameters
into a :class:`CalibrationLayout`. ``CalibrationCmd`` builds the layout
into its own layer and rebuilds it when the parameters change; the
parameters are kept in a hidden :class:`CalibrationRecord` asset.
"""

from .focus_test import FocusMode, FocusTestGenerator
from .generator import CalibrationGenerator
from .interval_test import IntervalTestGenerator
from .layout import (
    CalibrationLayout,
    CellEntry,
    CommandEntry,
    FilledRect,
    ProceduralShape,
    VectorShape,
)
from .material_test_grid import MaterialTestGridGenerator
from .record import CalibrationRecord, CalibrationState

#: The layer type of the layer a test is generated on. Tests own their
#: layer, so other workpieces cannot be placed in it.
LAYER_TYPE_ID = "calibration_test"

GENERATORS: dict[str, CalibrationGenerator] = {
    generator.id: generator
    for generator in (
        MaterialTestGridGenerator(),
        IntervalTestGenerator(),
        FocusTestGenerator(),
    )
}

__all__ = [
    "GENERATORS",
    "LAYER_TYPE_ID",
    "CalibrationGenerator",
    "CalibrationLayout",
    "CalibrationRecord",
    "CalibrationState",
    "CellEntry",
    "CommandEntry",
    "FilledRect",
    "FocusMode",
    "FocusTestGenerator",
    "IntervalTestGenerator",
    "MaterialTestGridGenerator",
    "ProceduralShape",
    "VectorShape",
]
