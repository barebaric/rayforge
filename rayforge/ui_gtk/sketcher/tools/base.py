from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import cairo

if TYPE_CHECKING:
    from ....core.sketcher.commands.base import PreviewState
    from ..sketchelement import SketchElement


class SketcherKey(Enum):
    """UI-agnostic special key identifiers."""

    BACKSPACE = auto()
    DELETE = auto()
    ARROW_LEFT = auto()
    ARROW_RIGHT = auto()
    RETURN = auto()
    ESCAPE = auto()
    HOME = auto()
    END = auto()
    UNDO = auto()
    REDO = auto()
    COPY = auto()
    PASTE = auto()
    CUT = auto()
    SELECT_ALL = auto()


class SketchTool(ABC):
    """Abstract base class for sketcher tools."""

    SHORTCUT: Optional[Tuple[str, str]] = None

    def __init__(self, element: SketchElement):
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

    def get_preview_state(self) -> Optional["PreviewState"]:
        """
        Returns the current preview state for tools that support live preview.
        Override in subclasses that have a _preview_state attribute.
        """
        return None

    def handle_text_input(self, text: str) -> bool:
        """Optional hook for handling printable character input."""
        return False

    def handle_key_event(self, key: SketcherKey) -> bool:
        """Optional hook for handling special (non-character) key events."""
        return False

    def get_active_shortcuts(
        self,
    ) -> List[Tuple[Union[str, List[str]], str, Optional[Callable[[], bool]]]]:
        """
        Returns shortcuts currently available based on tool state.

        Returns a list of (key, label, condition) tuples.
        - key: Either a string (single key) or list of strings (multiple keys)
        - label: Human-readable description
        - condition: Optional callable returning True if shortcut should be
                     shown. If None, shortcut is always shown.

        Override in subclasses to provide context-sensitive shortcuts.
        """
        return []
