from abc import ABC, abstractmethod
import cairo


class SketchTool(ABC):
    """Abstract base class for sketcher tools."""

    def __init__(self, element):
        self.element = element

    @abstractmethod
    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        pass

    @abstractmethod
    def on_drag(self, world_dx: float, world_dy: float):
        pass

    @abstractmethod
    def on_release(self, world_x: float, world_y: float):
        pass

    def on_hover_motion(self, world_x: float, world_y: float):
        """Optional hook for hover effects."""
        pass

    def on_deactivate(self):
        """
        Called when the tool is about to be switched or deactivated.
        Subclasses can implement this to clean up their state.
        """
        pass

    def draw_overlay(self, ctx: cairo.Context):
        """
        Called by the SketchElement to allow the active tool to draw
        transient UI (like selection boxes) in screen space.
        """
        pass
