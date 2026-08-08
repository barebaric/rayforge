"""
Document/model signal hub for the 3D canvas.

Owns the machine and document signal subscriptions, the viewport (re)build
on WCS/layer changes, and the active-layer WCS tracking.  The canvas asks
the hub to refresh the viewport through callbacks and keeps scene
compilation out of this module.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor
    from ...machine.models.machine import Machine
    from .viewport import ViewportConfig


class DocSignalHub:
    """
    Manages the machine/doc signal wiring and WCS viewport math.

    Signal handlers that need scene state are provided as callbacks so the
    hub stays a pure subscription/viewport module.
    """

    def __init__(
        self,
        context,
        doc_editor: "DocEditor",
        *,
        set_viewport: Callable[["ViewportConfig"], None],
        mark_scene_dirty: Callable[[], None],
        request_render: Callable[[], None],
        refresh_scene: Callable[[], None],
        get_gl_initialized: Callable[[], bool],
    ):
        self._context = context
        self._doc_editor = doc_editor
        self._set_viewport = set_viewport
        self._mark_scene_dirty = mark_scene_dirty
        self._request_render = request_render
        self._refresh_scene = refresh_scene
        self._get_gl_initialized = get_gl_initialized

        self._active_layer_wcs_conn = None

    @property
    def doc(self) -> "Doc":
        """Returns the current document from the editor."""
        return self._doc_editor.doc

    @property
    def pipeline(self):
        """Returns the current pipeline from the editor."""
        return self._doc_editor.pipeline

    @property
    def rotary_enabled(self) -> bool:
        """Returns True if the active layer has rotary mode enabled."""
        if self.doc and self.doc.active_layer:
            return self.doc.active_layer.rotary_enabled
        return False

    def connect(self):
        """Subscribe to the machine and doc signals."""
        machine = self._context.machine
        if machine:
            machine.wcs_updated.connect(self._on_wcs_updated)
            machine.changed.connect(self._on_wcs_updated)
            self._on_wcs_updated(machine)

        self.doc.active_layer_changed.connect(self._on_active_layer_changed)
        self._connect_active_layer_wcs()

    def disconnect(self):
        """Unsubscribe from the machine and doc signals."""
        self._disconnect_active_layer_wcs()
        self.doc.active_layer_changed.disconnect(self._on_active_layer_changed)

        machine = self._context.machine
        if machine:
            machine.wcs_updated.disconnect(self._on_wcs_updated)
            machine.changed.disconnect(self._on_wcs_updated)

    def set_machine(self, viewport: Optional["ViewportConfig"] = None):
        """Reconnect the machine signals and refresh the viewport."""
        old_machine = self._context.machine
        if old_machine:
            old_machine.wcs_updated.disconnect(self._on_wcs_updated)
            old_machine.changed.disconnect(self._on_wcs_updated)

        if viewport is None:
            from .viewport import ViewportConfig

            viewport = ViewportConfig.default()

        self._set_viewport(viewport)

        new_machine = self._context.machine
        if new_machine:
            new_machine.wcs_updated.connect(self._on_wcs_updated)
            new_machine.changed.connect(self._on_wcs_updated)
            self._on_wcs_updated(new_machine)

        if self._get_gl_initialized():
            self._refresh_scene()

    def _on_wcs_updated(self, machine: "Machine", **kwargs):
        """Handler for when the machine's WCS state changes."""
        if machine:
            self._set_viewport(self._build_viewport(machine))
        self._mark_scene_dirty()
        self._request_render()

    def _get_active_layer_wcs_offset(self, machine: "Machine"):
        """Returns the WCS offset for the active layer."""
        layer = self.doc.active_layer if self.doc else None
        if layer and layer.wcs:
            return machine.get_wcs_offset(layer.wcs)
        return machine.get_active_wcs_offset()

    def _build_viewport(self, machine: "Machine") -> "ViewportConfig":
        """Build a ViewportConfig using the active layer's WCS."""
        from .viewport import ViewportConfig

        return ViewportConfig.from_machine_with_wcs(
            machine, self._get_active_layer_wcs_offset(machine)
        )

    def _connect_active_layer_wcs(self):
        """Connect to the active layer's updated signal for WCS changes."""
        self._disconnect_active_layer_wcs()

        layer = self.doc.active_layer
        if layer:
            self._active_layer_wcs_conn = layer.updated.connect(
                self._on_active_layer_updated
            )

    def _disconnect_active_layer_wcs(self):
        """Disconnect the active layer's updated signal."""
        if self._active_layer_wcs_conn is not None:
            old_layer = self.doc.active_layer
            old_layer.updated.disconnect(self._active_layer_wcs_conn)
            self._active_layer_wcs_conn = None

    def _on_active_layer_changed(self, sender):
        """Reconnect WCS tracking to the new active layer."""
        self._connect_active_layer_wcs()
        machine = self._context.machine
        if machine:
            self._on_wcs_updated(machine)

    def _on_active_layer_updated(self, layer):
        """Handle property changes on the active layer, including WCS."""
        machine = self._context.machine
        if machine:
            self._on_wcs_updated(machine)
