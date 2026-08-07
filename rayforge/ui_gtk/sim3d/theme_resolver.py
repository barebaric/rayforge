"""
Theme + colour resolution for the 3D canvas.

Delegates the shared domain colours (base ``ColorSet``, laser colour
sets) to the context-wide :class:`ThemeColorService`, and keeps only the
GL-specific background/axis/grid derivation local.  The canvas asks the
resolver to refresh when its theme is dirty and reads the resolved state
through properties.
"""

from typing import TYPE_CHECKING, Callable, Optional

from OpenGL import GL

from ...context import get_context
from ...core.color import ColorSet

if TYPE_CHECKING:
    from gi.repository import Gtk

    from ...machine.models.machine import Machine
    from .renderer.scene_renderer import SceneRenderer


class ThemeResolver:
    """
    Resolves theme-derived colours and colour LUTs for the 3D scene.

    Base ``ColorSet`` and laser colour sets come from the shared theme
    service; the GL background/axis/grid derivation stays here.
    """

    def __init__(
        self,
        widget: "Gtk.Widget",
        scene: "SceneRenderer",
        get_machine: Callable[[], Optional["Machine"]],
        get_gl_initialized: Callable[[], bool],
        request_render: Callable[[], None],
    ):
        self._widget = widget
        self._scene = scene
        self._get_machine = get_machine
        self._get_gl_initialized = get_gl_initialized
        self._request_render = request_render

    @property
    def color_set(self) -> Optional[ColorSet]:
        """The resolved theme ColorSet, or None if not yet resolved."""
        return get_context().theme.color_set

    @property
    def theme_is_dirty(self) -> bool:
        """True if theme-derived colours need re-resolving."""
        return get_context().theme.dirty

    def mark_dirty(self):
        """Mark theme-derived colours as needing re-resolution."""
        get_context().theme.mark_dirty()

    def on_style_changed(self, widget, gparam):
        """Marks theme resources as dirty when the GTK theme changes."""
        get_context().theme.mark_dirty()
        self._request_render()

    def update_theme_and_colors(self):
        """
        Resolves the ColorSet and updates other theme-dependent elements.
        """
        if not self._scene.axis_renderer or not self._scene.texture_renderer:
            return

        service = get_context().theme
        service.set_machine(self._get_machine())
        color_set = service.color_set
        if color_set is None:
            return

        style_context = self._widget.get_style_context()
        found, bg_rgba = style_context.lookup_color("theme_bg_color")
        if not found:
            found, bg_rgba = style_context.lookup_color("view_bg_color")

        if found:
            bg_color = (
                bg_rgba.red * 0.35,
                bg_rgba.green * 0.35,
                bg_rgba.blue * 0.35,
            )
            bg_light = (
                min(1.0, bg_rgba.red * 0.9),
                min(1.0, bg_rgba.green * 0.9),
                min(1.0, bg_rgba.blue * 0.9),
            )
            clear_color = (
                bg_rgba.red,
                bg_rgba.green,
                bg_rgba.blue,
                bg_rgba.alpha,
            )
        else:
            bg_color = (0.11, 0.11, 0.14)
            bg_light = (0.2, 0.2, 0.25)
            clear_color = (0.2, 0.2, 0.25, 1.0)

        self._scene.apply_background_colors(bg_color, bg_light)

        GL.glClearColor(*clear_color)

        # Get the foreground color for axes and labels
        found, fg_rgba = style_context.lookup_color("view_fg_color")
        if found:
            axis_color = (
                fg_rgba.red,
                fg_rgba.green,
                fg_rgba.blue,
                fg_rgba.alpha,
            )
            # Grid color is derived from fg color to be less prominent
            grid_color = fg_rgba.red, fg_rgba.green, fg_rgba.blue, 0.5
            bg_plane_color = fg_rgba.red, fg_rgba.green, fg_rgba.blue, 0.08

            self._scene.apply_axis_colors(
                axis_color, grid_color, bg_plane_color
            )

        self.update_renderer_color_luts()

    def update_renderer_color_luts(self):
        if not self._get_gl_initialized():
            return

        provider = get_context().theme.color_lut_provider()
        if provider is None:
            return

        self._scene.update_color_luts(provider)
