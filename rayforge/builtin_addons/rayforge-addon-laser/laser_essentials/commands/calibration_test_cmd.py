"""
Creates Interval Tests and Focus Tests as ordinary project content.

Each test becomes its own layer. Every cell or line is a workpiece
with its own step, linked through ``Step.generated_workpiece_uid`` the
same way the Material Test Grid links its step, so each one keeps its
own settings and stays editable in the step settings. Labels share one
workpiece with one low-power contour step. The Focus Test puts Command
steps (``M0`` pauses or relative Z moves) between its lines.
"""

import logging
import re
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Protocol, cast

from raygeo.geo import Geometry, Matrix
from raygeo.geo.shape.text import FontConfig, text_to_geometry

from rayforge.core.layer import Layer
from rayforge.core.source_asset import SourceAsset
from rayforge.core.step import Step
from rayforge.core.step_registry import step_registry
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.layer_cmd import AddLayerAndSetActiveCommand
from rayforge.image.svg.exporter import GeometrySvgExporter
from rayforge.image.svg.importer import SvgImporter
from rayforge.pipeline.intent_builder import _driver_uses_gcode
from rayforge.pipeline.transformer.registry import transformer_registry

from ..calibration_tests import (
    FocusMode,
    FocusTestParams,
    IntervalTestParams,
    focus_commands,
    focus_offsets,
    format_offset,
    interval_values,
    lines_per_inch,
    row_positions,
)
from ..steps import ContourStep, EngraveStep
from ..steps.laser_step import LaserStep

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    class OverscanTransformerType(Protocol):
        @staticmethod
        def calculate_auto_distance(
            step_speed: int, max_acceleration: int
        ) -> float: ...


logger = logging.getLogger(__name__)

# Gap between a cell or line and its label, in label heights.
_LABEL_GAP = 0.6
# Spacing of the distance ticks along the Focus Test ramp, in mm.
_RAMP_TICK_MM = 10.0
_RAMP_TICK_LENGTH = 2.0


def _rect(x: float, y: float, w: float, h: float) -> Geometry:
    geo = Geometry()
    geo.move_to(x, y)
    geo.line_to(x + w, y)
    geo.line_to(x + w, y + h)
    geo.line_to(x, y + h)
    geo.close_path()
    return geo


def _line(x0: float, y0: float, x1: float, y1: float) -> Geometry:
    geo = Geometry()
    geo.move_to(x0, y0)
    geo.line_to(x1, y1)
    return geo


def _text(
    text: str,
    height: float,
    center_x: float,
    top: float,
    max_width: float | None = None,
) -> Geometry:
    """
    Glyph outlines of *text*, *height* mm tall and centered below *top*,
    scaled down further when needed to fit in *max_width*.
    """
    geo = text_to_geometry(text, FontConfig(size=10.0))
    min_x, min_y, max_x, max_y = geo.rect()
    scale = height / max(max_y - min_y, 1e-9)
    if max_width is not None and (max_x - min_x) * scale > max_width:
        scale = max_width / max(max_x - min_x, 1e-9)
    width = (max_x - min_x) * scale
    text_height = (max_y - min_y) * scale
    matrix = (
        Matrix.translation(center_x - width / 2, top - text_height)
        @ Matrix.scale(scale, scale)
        @ Matrix.translation(-min_x, -min_y)
    )
    geo.transform(matrix)
    return geo


class CalibrationTestCmd:
    """Creates Interval Tests and Focus Tests in the document."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self._pending_assets: list[SourceAsset] = []

    # -- Availability -----------------------------------------------------

    def available_focus_modes(self) -> list[FocusMode]:
        """
        The Focus Test modes the current machine supports: Z axis steps
        need a Z axis, and both Z and manual mode need a G-code driver
        and the Command step (Automation addon).
        """
        modes = [FocusMode.RAMP]
        machine = self._editor.context.machine
        if machine is None or step_registry.get("CommandStep") is None:
            return modes
        if not _driver_uses_gcode(machine):
            return modes
        modes.insert(0, FocusMode.MANUAL)
        if machine.has_z_axis:
            modes.insert(0, FocusMode.Z_AXIS)
        return modes

    # -- Interval Test ----------------------------------------------------

    def create_interval_test(self, params: IntervalTestParams) -> Layer:
        """Adds an Interval Test layer and returns it."""
        intervals = interval_values(params)
        layer = Layer(_("Interval Test"))
        size = params.cell_size
        xs = row_positions(len(intervals), size, params.spacing)
        cell_y = self._label_block_height(params, lines=2)
        if params.include_labels:
            labels = Geometry()
            for x, interval in zip(xs, intervals):
                center = x + size / 2
                top = cell_y - _LABEL_GAP * params.label_height
                fit = (size + params.spacing) * 0.9
                labels.extend(
                    _text(
                        f"{interval:.2f}",
                        params.label_height,
                        center,
                        top,
                        fit,
                    )
                )
                labels.extend(
                    _text(
                        _("{lpi:.0f} LPI").format(
                            lpi=lines_per_inch(interval)
                        ),
                        params.label_height,
                        center,
                        top - params.label_height * 1.5,
                        fit,
                    )
                )
            self._add_labels(layer, labels, params)
        for x, interval in zip(xs, intervals):
            name = _("Interval {interval:.2f} mm").format(interval=interval)
            wp = self._add_workpiece(
                layer, _rect(x, cell_y, size, size), name, filled=True
            )
            step = self._engrave_step(params.power_percent, params.speed)
            step.name = name
            step.line_interval_mm = interval
            step.depth_mode = "CONSTANT_POWER"
            self._link(layer, step, wp)
        self._insert(layer, _("Add Interval Test"))
        return layer

    # -- Focus Test -------------------------------------------------------

    def create_focus_test(self, params: FocusTestParams) -> Layer:
        """Adds a Focus Test layer and returns it."""
        if params.mode not in self.available_focus_modes():
            raise ValueError(
                _(
                    "This machine does not support the focus test mode {mode}."
                ).format(mode=params.mode.label)
            )
        layer = Layer(_("Focus Test"))
        if params.mode == FocusMode.RAMP:
            self._build_ramp(layer, params)
        else:
            self._build_focus_lines(layer, params)
        self._insert(layer, _("Add Focus Test"))
        return layer

    def _build_focus_lines(self, layer: Layer, params: FocusTestParams):
        offsets = focus_offsets(params)
        xs = row_positions(len(offsets), 0.0, params.spacing)
        line_y = self._label_block_height(params, lines=2)
        if params.include_labels:
            # Labels alternate between two rows, so each may be almost
            # twice as wide as the line spacing.
            labels = Geometry()
            top = line_y - _LABEL_GAP * params.label_height
            for index, (x, offset) in enumerate(zip(xs, offsets)):
                row_top = top - (index % 2) * params.label_height * 1.5
                labels.extend(
                    _text(
                        format_offset(offset),
                        params.label_height,
                        x,
                        row_top,
                        params.spacing * 1.8,
                    )
                )
            self._add_labels(layer, labels, params)
        machine = self._editor.context.machine
        reverse_z = bool(machine and machine.reverse_z_axis)
        commands: dict[int, tuple[list[str], str]] = {
            index: (lines, name)
            for index, lines, name in focus_commands(params, reverse_z)
        }
        for index, (x, offset) in enumerate(zip(xs, offsets)):
            self._add_command(layer, commands.get(index))
            name = _("Focus {offset} mm").format(offset=format_offset(offset))
            wp = self._add_workpiece(
                layer,
                _line(x, line_y, x, line_y + params.line_length),
                name,
                filled=False,
            )
            step = self._contour_step(params.power_percent, params.speed)
            step.name = name
            self._link(layer, step, wp)
        self._add_command(layer, commands.get(len(offsets)))

    def _build_ramp(self, layer: Layer, params: FocusTestParams):
        length = params.ramp_length
        line_y = self._label_block_height(params, lines=1)
        line_y += _RAMP_TICK_LENGTH
        if params.include_labels:
            labels = Geometry()
            ticks = int(length // _RAMP_TICK_MM) + 1
            for i in range(ticks):
                x = i * _RAMP_TICK_MM
                labels.extend(
                    _line(x, line_y - _RAMP_TICK_LENGTH, x, line_y - 0.5)
                )
                labels.extend(
                    _text(
                        f"{x:g}",
                        params.label_height,
                        x,
                        line_y - _RAMP_TICK_LENGTH - 0.5,
                    )
                )
            self._add_labels(layer, labels, params)
        name = _("Focus ramp")
        wp = self._add_workpiece(
            layer, _line(0.0, line_y, length, line_y), name, filled=False
        )
        step = self._contour_step(params.power_percent, params.speed)
        step.name = name
        self._link(layer, step, wp)

    # -- Building blocks --------------------------------------------------

    @staticmethod
    def _label_block_height(params, lines: int) -> float:
        """Height reserved below the cells for *lines* lines of labels."""
        if not params.include_labels:
            return 0.0
        height = params.label_height
        return height * (_LABEL_GAP + lines + (lines - 1) * 0.5)

    def _add_labels(self, layer: Layer, labels: Geometry, params):
        wp = self._add_workpiece(layer, labels, _("Labels"), filled=False)
        step = self._contour_step(
            params.label_power_percent, params.label_speed
        )
        step.name = _("Labels")
        self._link(layer, step, wp)

    def _add_command(
        self, layer: Layer, command: tuple[list[str], str] | None
    ):
        if command is None:
            return
        lines, name = command
        step_cls: Any = step_registry.get("CommandStep")
        assert step_cls is not None
        step = step_cls.create(self._editor.context, name=name)
        step.set_command_text("\n".join(lines))
        assert layer.workflow is not None
        layer.workflow.add_step(step)

    def _contour_step(self, power_percent: float, speed: float) -> Step:
        step = ContourStep.create(self._editor.context)
        self._set_laser(step, power_percent, speed)
        return step

    def _engrave_step(self, power_percent: float, speed: float) -> EngraveStep:
        step = EngraveStep.create(self._editor.context)
        self._set_laser(step, power_percent, speed)
        self._update_overscan(step)
        return step

    @staticmethod
    def _set_laser(step: LaserStep, power_percent: float, speed: float):
        step.set_power(power_percent / 100.0)
        step.set_cut_speed(int(speed))

    def _update_overscan(self, step: Step):
        """Recomputes the automatic overscan for the step's speed."""
        machine = self._editor.context.machine
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

    @staticmethod
    def _link(layer: Layer, step: Step, wp: WorkPiece):
        step.generated_workpiece_uid = wp.uid
        assert layer.workflow is not None
        layer.workflow.add_step(step)

    def _add_workpiece(
        self, layer: Layer, geometry: Geometry, name: str, filled: bool
    ) -> WorkPiece:
        """
        Imports *geometry* as a vector workpiece at its own coordinates
        (relative to the test's bottom-left corner).
        """
        svg = GeometrySvgExporter(geometry).export().decode("utf-8")
        if filled:
            svg = re.sub(r'fill="none"', 'fill="black"', svg)
            svg = re.sub(r'stroke="black"', 'stroke="none"', svg)
        result = SvgImporter(svg.encode("utf-8")).get_doc_items(
            PassthroughSpec()
        )
        if not result or not result.payload or not result.payload.items:
            raise RuntimeError(f"Could not create the {name} workpiece.")
        source: SourceAsset = result.payload.source
        source.name = name
        wp = result.payload.items[0]
        assert isinstance(wp, WorkPiece)
        wp.name = name
        world = wp.get_world_geometry()
        assert world is not None
        min_x, min_y, _max_x, _max_y = world.rect()
        target_x, target_y, _tx, _ty = geometry.rect()
        wp.pos = (
            wp.pos[0] + target_x - min_x,
            wp.pos[1] + target_y - min_y,
        )
        layer.add_child(wp)
        self._pending_assets.append(source)
        return wp

    def _insert(self, layer: Layer, undo_name: str):
        """Centers the test on the work area and adds it undoably."""
        rects = [
            wp.get_world_geometry().rect()  # type: ignore[union-attr]
            for wp in layer.all_workpieces
        ]
        min_x = min(r[0] for r in rects)
        min_y = min(r[1] for r in rects)
        max_x = max(r[2] for r in rects)
        max_y = max(r[3] for r in rects)
        dims = self._editor.machine_dimensions
        if dims:
            dx = dims[0] / 2 - (min_x + max_x) / 2
            dy = dims[1] / 2 - (min_y + max_y) / 2
            for wp in layer.all_workpieces:
                wp.pos = (wp.pos[0] + dx, wp.pos[1] + dy)
        doc = self._editor.doc
        for asset in self._pending_assets:
            doc.add_asset(asset)
        self._pending_assets = []
        with self._editor.history_manager.transaction(undo_name) as t:
            t.execute(
                AddLayerAndSetActiveCommand(
                    self._editor, layer, name=undo_name
                )
            )
        logger.info("Created %s", layer.name)
