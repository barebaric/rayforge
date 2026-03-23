import cairo
import logging
from blinker import Signal
from typing import List, Optional, TYPE_CHECKING, cast

from rayforge.core.matrix import Matrix
from rayforge.ui_gtk.canvas import CanvasElement
from ..core.sketch import Sketch
from ..core.entities import Line
from ..core.selection import SketchSelection
from ..core.types import EntityID
from .hittest import SketchHitTester
from .renderer import SketchRenderer
from .tools import TOOL_REGISTRY

if TYPE_CHECKING:
    from rayforge.core.undo.command import Command

    from .editor import SketchEditor
    from .sketchcanvas import SketchCanvas

logger = logging.getLogger(__name__)


class SketchElement(CanvasElement):
    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        width: float = 1.0,
        height: float = 1.0,
        sketch: Optional[Sketch] = None,
        **kwargs,
    ):
        # Pass the required positional arguments to the parent class.
        super().__init__(
            x=x,
            y=y,
            width=width,
            height=height,
            is_editable=True,
            clip=False,
            **kwargs,
        )

        # Signals
        self.constraint_edit_requested = Signal()
        self.tool_changed = Signal()
        self.solved = Signal()

        # Model
        self._sketch: Sketch
        self.external_hovered_constraint_idx: Optional[int] = None

        # State Managers
        self.selection = SketchSelection()
        self.hittester = SketchHitTester()
        self.renderer = SketchRenderer(self)
        self.editor: Optional["SketchEditor"]

        # This must be set after self.selection is initialized
        self.sketch = sketch if sketch is not None else Sketch()

        # Tools
        self.tools = {
            name: tool_cls(self) for name, tool_cls in TOOL_REGISTRY.items()
        }
        self.active_tool_name = "select"

        # Config
        self.point_radius = 5.0
        self.line_width = 2.0

        # Visibility toggles
        self.show_constraints = True
        self.show_construction_geometry = True

    @property
    def sketch(self) -> Sketch:
        return self._sketch

    @sketch.setter
    def sketch(self, new_sketch: Sketch):
        logger.debug(f"Called for sketch '{new_sketch.name}'")
        # Disconnect from old sketch's signals if it exists
        if hasattr(self, "_sketch"):
            self._disconnect_signals()

        self._sketch = new_sketch

        # Connect to new sketch's signals
        self._connect_signals()

    def _connect_signals(self):
        """Connects to signals that indicate the model has changed."""
        self.sketch.updated.connect(self._on_model_changed)
        if self.sketch and self.sketch.input_parameters is not None:
            logger.debug(
                f"Connecting to VarSet signals on "
                f"{type(self.sketch.input_parameters).__name__} "
                f"(id: {id(self.sketch.input_parameters)})"
            )
            self.sketch.input_parameters.var_added.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.var_removed.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.var_value_changed.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.var_definition_changed.connect(
                self._on_model_changed
            )
            self.sketch.input_parameters.cleared.connect(
                self._on_model_changed
            )

    def _disconnect_signals(self):
        """Disconnects signals to prevent leaks."""
        self.sketch.updated.disconnect(self._on_model_changed)
        if self.sketch and self.sketch.input_parameters is not None:
            logger.debug(
                "Disconnecting from VarSet signals on "
                f"{type(self.sketch.input_parameters).__name__} "
                f"(id: {id(self.sketch.input_parameters)})"
            )
            try:
                self.sketch.input_parameters.var_added.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.var_removed.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.var_value_changed.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.var_definition_changed.disconnect(
                    self._on_model_changed
                )
                self.sketch.input_parameters.cleared.disconnect(
                    self._on_model_changed
                )
            except Exception as e:
                logger.warning(
                    f"Error during signal disconnection (safe to ignore): {e}"
                )

    def _on_model_changed(self, sender, **kwargs):
        """
        Central handler for all model changes. Triggers a solve and redraw.
        """
        logger.debug(
            f"Triggered by {type(sender).__name__} (id: {id(sender)}) "
            f"with kwargs: {kwargs}. Solving and redrawing."
        )
        self.sketch.solve()
        self.update_bounds_from_sketch()
        self.mark_dirty()
        self.solved.send(self)

    def remove(self):
        """Overrides remove to cleanup signal connections."""
        self._disconnect_signals()
        super().remove()

    @property
    def current_tool(self):
        return self.tools.get(self.active_tool_name, self.tools["select"])

    def execute_command(self, command: "Command"):
        """Executes a command via the history manager if available."""
        if self.editor:
            self.editor.history_manager.execute(command)

    def get_selected_elements(self) -> bool:
        """
        Helper method to check if any internal items (points, entities, etc.)
        are selected. Returns a boolean, not a list of elements.
        """
        sel = self.selection
        return bool(
            sel.point_ids
            or sel.entity_ids
            or sel.constraint_idx is not None
            or sel.junction_pid is not None
        )

    def unselect_all(self):
        """Clears the internal sketch selection."""
        self.selection.clear()
        self.mark_dirty()

    def update_bounds_from_sketch(self):
        """
        Calculates the bounding box of the sketch geometry and updates the
        element's size and transform. For empty sketches, it creates a
        minimum-sized box and centers the origin. For non-empty sketches,
        it shrinks to fit the geometry exactly.
        """
        # A sketch is considered "empty" for bounding purposes if it has no
        # entities and at most one point (which would be the origin).
        is_truly_empty = (
            len(self.sketch.registry.entities) == 0
            and len(self.sketch.registry.points) <= 1
        )

        new_width: float
        new_height: float
        new_offset_x: float
        new_offset_y: float

        if is_truly_empty:
            # Apply a minimum dimension for selectability and center the
            # origin.
            min_dim = 50.0
            new_width = min_dim
            new_height = min_dim
            new_offset_x = min_dim / 2.0
            new_offset_y = min_dim / 2.0
        else:
            # Calculate the precise bounding box of all geometry.
            geometry = self.sketch.to_geometry()
            if geometry.is_empty():
                # This case handles sketches with only points.
                if not self.sketch.registry.points:
                    min_x, max_x, min_y, max_y = 0, 0, 0, 0
                else:
                    xs = [p.x for p in self.sketch.registry.points]
                    ys = [p.y for p in self.sketch.registry.points]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
            else:
                min_x, min_y, max_x, max_y = geometry.rect()

            # The element size is exactly the geometry size. No padding.
            new_width = max_x - min_x
            new_height = max_y - min_y
            # The offset moves the geometry's top-left to the element's origin.
            new_offset_x = -min_x
            new_offset_y = -min_y

        # Calculate the change in offset needed to keep the content visually
        # stationary on the canvas during the bounds update.
        current_offset_x, current_offset_y = (
            self.content_transform.get_translation()
        )
        delta_x = new_offset_x - current_offset_x
        delta_y = new_offset_y - current_offset_y

        # Apply all the calculated updates.
        self.content_transform = Matrix.translation(new_offset_x, new_offset_y)
        self.width = new_width
        self.height = new_height

        # Update the element's main transform to counteract the content shift.
        self.set_transform(
            self.transform @ Matrix.translation(-delta_x, -delta_y)
        )

        self.mark_dirty()

    def on_edit_mode_leave(self):
        """Called when this element is no longer the Canvas's edit_context."""
        self.update_bounds_from_sketch()

    # =========================================================================
    # Rendering
    # =========================================================================

    def draw(self, ctx: cairo.Context):
        """Main draw entry point."""
        self.renderer.draw(ctx)

    def draw_edit_overlay(self, ctx: cairo.Context):
        """Draws constraints, points, and handles on top of the canvas."""
        self.renderer.draw_edit_overlay(ctx)
        # Allow the active tool to draw its own overlay (e.g. selection box)
        self.current_tool.draw_overlay(ctx)

    # =========================================================================
    # Input Handling (Routed to Tools)
    # =========================================================================

    def handle_edit_press(
        self, world_x: float, world_y: float, n_press: int = 1
    ) -> bool:
        return self.current_tool.on_press(world_x, world_y, n_press)

    def handle_edit_drag(self, world_dx: float, world_dy: float):
        self.current_tool.on_drag(world_dx, world_dy)

    def handle_edit_release(self, world_x: float, world_y: float):
        self.current_tool.on_release(world_x, world_y)

    def on_hover_motion(self, world_x: float, world_y: float):
        """Dispatches hover events to the currently active tool."""
        self.current_tool.on_hover_motion(world_x, world_y)

    def get_lines_at_point(self, pid: EntityID) -> List[Line]:
        return [
            e
            for e in self.sketch.registry.entities
            if isinstance(e, Line) and pid in (e.p1_idx, e.p2_idx)
        ]

    def set_tool(self, tool_name: str):
        if tool_name in self.tools and self.active_tool_name != tool_name:
            # Deactivate the old tool before switching to the new one.
            self.current_tool.on_deactivate()
            self.active_tool_name = tool_name
            self.mark_dirty()
            self.tool_changed.send(self, tool_name=tool_name)
            if self.canvas:
                canvas = cast("SketchCanvas", self.canvas)
                canvas.update_sketch_cursor()
            self.current_tool.on_activate()

    def delete_selection(self) -> bool:
        return self.tools["delete"]._delete_selection()

    def toggle_construction_on_selection(self):
        self.tools["construction"]._toggle_construction()

    def add_chamfer_action(self):
        self.tools["chamfer"]._add_chamfer()

    def add_fillet_action(self):
        self.tools["fillet"]._add_fillet()

    def is_action_supported(self, action: str) -> bool:
        tool = self.tools.get(action)
        if not tool:
            return False

        sel = self.selection
        target = None
        target_type = None

        if sel.junction_pid is not None:
            target = self.sketch.registry.get_point(sel.junction_pid)
            target_type = "junction"
        elif sel.point_ids:
            target = self.sketch.registry.get_point(sel.point_ids[0])
            target_type = "point"
        elif sel.entity_ids:
            target = self.sketch.registry.get_entity(sel.entity_ids[0])
            target_type = "entity"
        elif sel.constraint_idx is not None:
            if 0 <= sel.constraint_idx < len(self.sketch.constraints):
                target = self.sketch.constraints[sel.constraint_idx]
            target_type = "constraint"

        return tool.is_available(target, target_type)

    def add_alignment_constraint(self):
        self.tools["coincident"]._add_constraint()

    def remove_point_if_unused(self, pid: Optional[int]) -> bool:
        """
        Removes a point from the registry if it's not part of any entity.

        Args:
            pid: The point ID to remove.

        Returns:
            True if the point was removed, False otherwise.
        """
        if pid is None:
            return False
        removed = self.sketch.remove_point_if_unused(pid)
        if removed:
            self.mark_dirty()
        return removed

    def mark_dirty(self, ancestors=False, recursive=False):
        super().mark_dirty(ancestors=ancestors, recursive=recursive)
        if self.canvas:
            self.canvas.queue_draw()
