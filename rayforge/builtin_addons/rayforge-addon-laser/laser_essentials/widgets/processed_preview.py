"""Preview of the bitmap an Engrave step sends to the assembler."""

import logging
import threading
from collections.abc import Callable
from gettext import gettext as _
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from gi.repository import Gdk, Gio, GLib, Gtk

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.machine import Machine

logger = logging.getLogger(__name__)

#: Longest side of the texture shown in the dialog; the saved bitmap
#: keeps the full resolution.
MAX_DISPLAY_PX = 1024

#: Delay before a step change re-renders the preview.
REFRESH_DELAY_MS = 400


def _first_layer_workpiece(step: Any) -> "WorkPiece | None":
    layer = getattr(step, "layer", None)
    if layer is None:
        return None
    workpieces = layer.all_workpieces
    return workpieces[0] if workpieces else None


def _to_texture(image: np.ndarray) -> Gdk.Texture:
    height, width = image.shape
    scale = MAX_DISPLAY_PX / max(height, width)
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        height, width = image.shape
    data = np.ascontiguousarray(image)
    return Gdk.MemoryTexture.new(
        width,
        height,
        Gdk.MemoryFormat.G8,
        GLib.Bytes.new(data.tobytes()),
        width,
    )


class ProcessedImagePreview(Gtk.Box):
    """Shows the step's processed bitmap for one workpiece of its layer
    and saves it as PNG."""

    def __init__(
        self,
        step: Any,
        get_machine: Callable[[], "Machine | None"],
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self._step = step
        self._get_machine = get_machine
        self._get_workpiece: Callable[[], WorkPiece | None] = lambda: (
            _first_layer_workpiece(step)
        )
        self._generation = 0
        self._refresh_source = 0
        self._preview: np.ndarray | None = None

        self.picture = Gtk.Picture()
        self.picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.picture.set_can_shrink(True)
        self.picture.set_size_request(-1, 240)
        self.picture.add_css_class("card")
        self.append(self.picture)

        self.status_label = Gtk.Label(wrap=True, xalign=0)
        self.status_label.add_css_class("dim-label")
        self.append(self.status_label)

        self.save_button = Gtk.Button(label=_("Save Processed Bitmap…"))
        self.save_button.set_halign(Gtk.Align.END)
        self.save_button.set_sensitive(False)
        self.save_button.connect("clicked", self._on_save_clicked)
        self.append(self.save_button)

        step.updated.connect(self._on_step_updated)
        self.connect("destroy", self._on_destroy)

    @property
    def preview(self) -> np.ndarray | None:
        """The last rendered preview (uint8, dark = engraved)."""
        return self._preview

    def set_workpiece_provider(
        self, provider: Callable[[], "WorkPiece | None"]
    ) -> None:
        self._get_workpiece = provider

    def refresh(self, sync: bool = False) -> None:
        """Re-render the preview, in a worker thread unless ``sync``."""
        self._generation += 1
        generation = self._generation
        workpiece = self._get_workpiece()
        machine = self._get_machine()
        if workpiece is None or machine is None:
            self._apply(generation, None, _("No image on this layer."))
            return
        self._set_status(_("Updating preview…"))
        if sync:
            self._render(generation, machine, workpiece)
            return
        threading.Thread(
            target=self._render,
            args=(generation, machine, workpiece),
            daemon=True,
        ).start()

    def save_to(self, path: str | Path) -> bool:
        """Write the current preview to ``path`` as PNG."""
        if self._preview is None:
            return False
        return bool(cv2.imwrite(str(path), self._preview))

    def _render(
        self, generation: int, machine: "Machine", workpiece: "WorkPiece"
    ) -> None:
        try:
            image = self._step.render_processed_preview(machine, workpiece)
        except Exception:
            logger.exception("Processed image preview failed")
            image = None
        message = None if image is not None else _("Nothing to engrave.")
        if threading.current_thread() is threading.main_thread():
            self._apply(generation, image, message)
        else:
            GLib.idle_add(self._apply, generation, image, message)

    def _apply(
        self,
        generation: int,
        image: np.ndarray | None,
        message: str | None,
    ) -> bool:
        if generation != self._generation:
            return GLib.SOURCE_REMOVE
        self._preview = image
        if image is None:
            self.picture.set_paintable(None)
        else:
            self.picture.set_paintable(_to_texture(image))
        self.save_button.set_sensitive(image is not None)
        self._set_status(message)
        return GLib.SOURCE_REMOVE

    def _set_status(self, message: str | None) -> None:
        self.status_label.set_text(message or "")
        self.status_label.set_visible(bool(message))

    def _on_step_updated(self, *args) -> None:
        if self._refresh_source:
            GLib.source_remove(self._refresh_source)
        self._refresh_source = GLib.timeout_add(
            REFRESH_DELAY_MS, self._on_refresh_timeout
        )

    def _on_refresh_timeout(self) -> bool:
        self._refresh_source = 0
        self.refresh()
        return GLib.SOURCE_REMOVE

    def _on_destroy(self, *args) -> None:
        self._generation += 1
        if self._refresh_source:
            GLib.source_remove(self._refresh_source)
            self._refresh_source = 0
        self._step.updated.disconnect(self._on_step_updated)

    def _on_save_clicked(self, _button) -> None:
        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Save Processed Bitmap"))
        dialog.set_initial_name(f"{self._step.name or 'engrave'}.png")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        png_filter = Gtk.FileFilter()
        png_filter.set_name(_("PNG images"))
        png_filter.add_mime_type("image/png")
        filters.append(png_filter)
        dialog.set_filters(filters)
        dialog.set_default_filter(png_filter)
        root = self.get_root()
        parent = root if isinstance(root, Gtk.Window) else None
        dialog.save(parent, None, self._on_save_response)

    def _on_save_response(self, dialog: Gtk.FileDialog, result) -> None:
        try:
            gfile = dialog.save_finish(result)
        except GLib.Error:
            return
        path = gfile.get_path() if gfile else None
        if path and not self.save_to(path):
            self._set_status(_("Could not save the processed bitmap."))
