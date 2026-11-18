"""
The Test Settings section of the layer settings dialog.

It shows the settings of the calibration test that lives on the layer,
so the settings are found in the layer settings like every other layer
setting.
"""

import logging
from typing import TYPE_CHECKING

from gi.repository import Adw

from rayforge.core.layer import Layer
from rayforge.ui_gtk.doceditor.layer_settings_sections import (
    LayerSettingsSection,
)

from ..calibration.lookup import record_for_layer
from ..calibration.record import CalibrationRecord
from .calibration_settings import CalibrationSettingsController

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class CalibrationLayerSettingsSection(LayerSettingsSection):
    """Shows the settings of the test that lives on the layer."""

    def __init__(self):
        self._controller: CalibrationSettingsController | None = None

    def can_handle(self, layer: Layer, editor: "DocEditor") -> bool:
        return self._record_for(layer, editor) is not None

    def create_widgets(
        self, layer: Layer, editor: "DocEditor"
    ) -> list[Adw.PreferencesGroup]:
        record = self._record_for(layer, editor)
        assert record is not None
        self._controller = CalibrationSettingsController(editor, record)
        return self._controller.widgets()

    def close(self):
        if self._controller is not None:
            self._controller.close()
            self._controller = None

    @staticmethod
    def _record_for(
        layer: Layer, editor: "DocEditor"
    ) -> CalibrationRecord | None:
        return record_for_layer(editor.doc, layer)
