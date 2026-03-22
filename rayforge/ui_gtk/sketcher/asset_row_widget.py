import logging
from gettext import gettext as _

from gi.repository import Gdk

from ...core.sketcher.sketch import Sketch
from ..doceditor.asset_row_widget import BaseAssetRowWidget

logger = logging.getLogger(__name__)


class SketchAssetRowWidget(BaseAssetRowWidget):
    """A widget representing a single Sketch asset in a list."""

    def __init__(self, doc, asset, editor):
        super().__init__(doc, asset, editor)
        self._sketch: Sketch = asset
        self.add_css_class("sketch-asset-row")
        self._build_common_structure(
            show_edit_button=True,
            edit_tooltip=_("Edit this sketch"),
        )
        self._sketch.updated.connect(self.on_asset_changed)
        self.update_ui()

    def do_destroy(self):
        self._sketch.updated.disconnect(self.on_asset_changed)

    def on_asset_changed(self, sender, **kwargs):
        self.update_ui()

    def get_drag_content(self) -> Gdk.ContentProvider:
        """Provides the content for a drag operation."""
        logger.debug(
            "Providing drag content for sketch UID: %s",
            repr(self._sketch.uid),
        )
        return Gdk.ContentProvider.new_for_value(str(self._sketch.uid))

    def update_ui(self):
        super().update_ui()
        param_count = len(self._sketch.input_parameters)
        if param_count == 0:
            subtitle_text = _("No parameters")
        elif param_count == 1:
            subtitle_text = _("1 parameter")
        else:
            subtitle_text = _("{count} parameters").format(count=param_count)

        self.subtitle_label.set_label(subtitle_text)
        self.subtitle_label.set_tooltip_text(subtitle_text)
