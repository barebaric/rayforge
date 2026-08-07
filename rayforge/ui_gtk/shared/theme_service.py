"""
Shared theme colour service.

Owns the single source of truth for the domain colours that both the 2D
and 3D canvases need: the base ``ColorSet`` (``OPS_COLOR_SPEC``), the
per-laser colour sets, and the per-layer colour sets.  It binds to one
GTK widget (the main window), resolves the theme colours through that
widget's style context, and re-resolves lazily when the theme, machine,
or document changes.

The service is exposed as a lazy ``RayforgeContext.theme`` property so
GTK is never imported eagerly in headless/worker processes.
"""

import logging
from typing import TYPE_CHECKING, Dict, Optional

from ...core.color import OPS_COLOR_SPEC, ColorSet, hex_to_rgba
from ...image.util.srgb import create_lut_from_color
from ...machine.models.colors import OpsColorSet
from ...machine.models.laser import LaserHead
from .color_lut_provider import ColorLutProvider
from .gtk_color import GtkColorResolver

if TYPE_CHECKING:
    from gi.repository import Gtk

    from ...core.doc import Doc
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)


class ThemeColorService:
    """
    Resolves and caches the shared domain colours for all canvases.

    Base ``ColorSet``, per-laser colour sets, and per-layer colour sets
    are each resolved exactly once per theme change, keyed by a single
    dirty flag.  Both canvases read through this service so laser paths
    and textures stay identical.
    """

    def __init__(self):
        self._widget: Optional["Gtk.Widget"] = None
        self._machine: Optional["Machine"] = None
        self._doc: Optional["Doc"] = None

        self._dirty = True
        self._color_set: Optional[ColorSet] = None
        self._laser_color_sets: Dict[str, ColorSet] = {}
        self._layer_color_sets: Dict[str, ColorSet] = {}
        self._lut_provider: Optional[ColorLutProvider] = None

    def bind(self, widget: "Gtk.Widget"):
        """
        Bind the service to a widget and start reacting to theme changes.

        The widget's style context is representative of the whole window;
        both canvases are descendants of the bound widget.
        """
        if self._widget is widget:
            return
        self._widget = widget
        widget.connect("notify::style", self._on_style_changed)
        self.mark_dirty()

    def set_machine(self, machine: Optional["Machine"]):
        """Set the machine whose lasers colour the laser paths."""
        if machine is self._machine:
            return
        if self._machine is not None:
            self._machine.changed.disconnect(self._on_machine_changed)
        self._machine = machine
        if machine is not None:
            machine.changed.connect(self._on_machine_changed)
        self.mark_dirty()

    def set_doc(self, doc: Optional["Doc"]):
        """Set the document whose layers colour the layer paths."""
        if doc is self._doc:
            return
        if self._doc is not None:
            self._doc.descendant_updated.disconnect(self._on_doc_updated)
        self._doc = doc
        if doc is not None:
            doc.descendant_updated.connect(self._on_doc_updated)
        self.mark_dirty()

    def _on_style_changed(self, widget, gparam):
        self.mark_dirty()

    def _on_machine_changed(self, machine):
        self.mark_dirty()

    def _on_doc_updated(self, *args, **kwargs):
        self.mark_dirty()

    def mark_dirty(self):
        """Mark all cached colours as stale."""
        self._dirty = True
        self._lut_provider = None

    @property
    def dirty(self) -> bool:
        """True if cached colours need re-resolving."""
        return self._dirty

    @property
    def color_set(self) -> Optional[ColorSet]:
        """The resolved base theme ColorSet."""
        self._refresh_if_dirty()
        return self._color_set

    @property
    def laser_color_sets(self) -> Dict[str, ColorSet]:
        """Per-laser colour sets keyed by laser UID."""
        self._refresh_if_dirty()
        return self._laser_color_sets

    @property
    def layer_color_sets(self) -> Dict[str, ColorSet]:
        """Per-layer colour sets keyed by layer UID."""
        self._refresh_if_dirty()
        return self._layer_color_sets

    def color_lut_provider(self) -> Optional[ColorLutProvider]:
        """A provider over the current base + laser colour sets."""
        color_set = self.color_set
        if color_set is None:
            return None
        if self._lut_provider is None:
            self._lut_provider = ColorLutProvider(
                color_set, self.laser_color_sets
            )
        return self._lut_provider

    def _refresh_if_dirty(self):
        if not self._dirty:
            return
        if self._widget is None:
            return
        resolver = GtkColorResolver(self._widget)
        self._color_set = resolver.resolve(OPS_COLOR_SPEC)
        self._laser_color_sets = self._resolve_laser_color_sets()
        self._layer_color_sets = self._resolve_layer_color_sets()
        self._dirty = False

    def _resolve_laser_color_sets(self) -> Dict[str, ColorSet]:
        if self._color_set is None or self._machine is None:
            return {}
        laser_color_sets: Dict[str, ColorSet] = {}
        for laser in self._machine.heads:
            if not isinstance(laser, LaserHead):
                continue
            laser_color_set = OpsColorSet.from_laser(laser, self._color_set)
            laser_color_sets[laser.uid] = laser_color_set.to_color_set()
        return laser_color_sets

    def _resolve_layer_color_sets(self) -> Dict[str, ColorSet]:
        if self._color_set is None or self._doc is None:
            return {}
        layer_color_sets: Dict[str, ColorSet] = {}
        for layer in self._doc.layers:
            cut_rgba = hex_to_rgba(layer.color)
            cut_lut = create_lut_from_color(cut_rgba)
            data = {
                "cut": cut_lut,
                "engrave": cut_lut,
                "travel": self._color_set.get_rgba("travel"),
                "zero_power": self._color_set.get_rgba("zero_power"),
            }
            layer_color_sets[layer.uid] = ColorSet(_data=data)
        return layer_color_sets
