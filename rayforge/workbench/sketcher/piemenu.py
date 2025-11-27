import logging
from blinker import Signal
from gi.repository import Gtk

from rayforge.shared.ui.piemenu import PieMenu, PieMenuItem

logger = logging.getLogger(__name__)


class SketchPieMenu(PieMenu):
    """
    Subclass of PieMenu specifically for the SketcherCanvas.
    Maps menu item clicks to high-level signals.
    """

    def __init__(self, parent_widget: Gtk.Widget):
        super().__init__(parent_widget)

        # High-level Signals
        self.tool_selected = Signal()
        self.constraint_selected = Signal()
        self.action_triggered = Signal()

        # Tools
        item = PieMenuItem("drag-handle-symbolic", "Select", data="select")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("laser-path-symbolic", "Line", data="line")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("laps-symbolic", "Arc", data="arc")
        item.on_click.connect(self._on_tool_clicked, weak=False)
        self.add_item(item)

        # Actions
        item = PieMenuItem(
            "layer-symbolic", "Construction", data="construction"
        )
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem("delete-symbolic", "Delete", data="delete")
        item.on_click.connect(self._on_action_clicked, weak=False)
        self.add_item(item)

        # Constraints
        item = PieMenuItem(
            "tabs-equidistant-symbolic", "Distance", data="dist"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "align-vertical-center-symbolic", "Horizontal", data="horiz"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

        item = PieMenuItem(
            "align-horizontal-center-symbolic", "Vertical", data="vert"
        )
        item.on_click.connect(self._on_constraint_clicked, weak=False)
        self.add_item(item)

    def _on_tool_clicked(self, sender):
        """Handle tool selection signals."""
        if sender.data:
            logger.info(f"Emitting tool selection: {sender.data}")
            self.tool_selected.send(self, tool=sender.data)

    def _on_constraint_clicked(self, sender):
        """Handle constraint selection signals."""
        if sender.data:
            logger.info(f"Emitting constraint: {sender.data}")
            self.constraint_selected.send(self, constraint_type=sender.data)

    def _on_action_clicked(self, sender):
        """Handle generic action signals."""
        if sender.data:
            logger.info(f"Emitting action: {sender.data}")
            self.action_triggered.send(self, action=sender.data)
