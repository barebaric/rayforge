import logging
from gi.repository import Gtk
from typing import Optional, List, TYPE_CHECKING, Tuple
from gettext import gettext as _
from ...context import get_context
from ...core.group import Group
from ...core.item import DocItem
from ...core.matrix import Matrix
from ...core.stock import StockItem
from ...core.workpiece import WorkPiece
from ..shared.expander import Expander
from .property_providers import (
    PropertyProvider,
    property_provider_registry,
)

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


logger = logging.getLogger(__name__)


class DocItemPropertiesWidget(Gtk.Box):
    """
    An orchestrator widget that displays properties for selected document
    items.

    It composes its UI from a set of registered "Property Provider" components,
    each responsible for a specific aspect of an item (e.g., transformation,
    source file, tabs). It manages persistent widgets to avoid interrupting
    user edits.

    Providers with ``separate_group = True`` each get their own Expander
    card, visually separated from the main item properties card.
    """

    def __init__(
        self,
        editor: "DocEditor",
        items: Optional[List[DocItem]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, *args, **kwargs)

        self.editor = editor
        self.items: List[DocItem] = []

        self._main_expander = Expander()
        self._main_expander.set_expanded(True)
        self._main_expander.set_title(_("Item Properties"))
        self.append(self._main_expander)

        self._rows_container = Gtk.ListBox()
        self._rows_container.set_selection_mode(Gtk.SelectionMode.NONE)
        self._main_expander.set_child(self._rows_container)

        self.providers: List[PropertyProvider] = (
            property_provider_registry.create_instances()
        )
        self._provider_widget_map: List[
            Tuple[PropertyProvider, List[Gtk.Widget]]
        ] = []
        self._separate_groups: List[
            Tuple[PropertyProvider, List[Gtk.Widget], Expander]
        ] = []
        self._initialize_providers_ui()

        self._machine = None
        self._connect_signals()

        self.set_items(items)

    def _connect_signals(self):
        """Connect to config and machine signals."""
        config = get_context().config
        config.changed.connect(self._on_config_changed)
        self._connect_machine_signals()

    def _connect_machine_signals(self):
        """Connect to machine signals to update UI when WCS changes."""
        machine = get_context().machine
        if machine and machine != self._machine:
            if self._machine:
                self._machine.changed.disconnect(self._on_machine_changed)
            self._machine = machine
            machine.changed.connect(self._on_machine_changed)

    def _on_config_changed(self, config):
        """Handle config changes (including machine switching)."""
        self._connect_machine_signals()
        if self.items:
            self._update_ui()

    def _on_machine_changed(self, machine):
        """Handle machine changes (including WCS selection/offset changes)."""
        if self.items:
            self._update_ui()

    def _initialize_providers_ui(self):
        """
        Creates all widgets for all providers one time and adds them to the
        appropriate container in a hidden state.
        """
        for provider in self.providers:
            widgets = provider.create_widgets()
            if provider.separate_group:
                expander = Expander()
                expander.set_expanded(True)
                if provider.group_title:
                    expander.set_title(provider.group_title)
                container = Gtk.ListBox()
                container.set_selection_mode(Gtk.SelectionMode.NONE)
                expander.set_child(container)
                for widget in widgets:
                    widget.set_visible(False)
                    container.append(widget)
                self._separate_groups.append((provider, widgets, expander))
                expander.set_margin_top(6)
                self.append(expander)
            else:
                self._provider_widget_map.append((provider, widgets))
                for widget in widgets:
                    widget.set_visible(False)
                    self._rows_container.append(widget)

    def set_items(self, items: Optional[List[DocItem]]):
        """Sets the currently selected items and updates the UI."""
        for item in self.items:
            item.updated.disconnect(self._on_item_data_changed)
            item.transform_changed.disconnect(self._on_item_data_changed)

        self.items = items or []

        count = len(self.items)
        if count == 1:
            self._main_expander.set_subtitle(_("1 item selected"))
        elif count > 1:
            self._main_expander.set_subtitle(
                _("{count} items selected").format(count=count)
            )
        else:
            self._main_expander.set_subtitle("")

        for item in self.items:
            item.updated.connect(self._on_item_data_changed)
            item.transform_changed.connect(self._on_item_data_changed)

        self._update_ui()

    def _on_item_data_changed(
        self, item, *, old_matrix: Optional["Matrix"] = None
    ):
        """
        Handles data changes from the DocItem model by updating the UI to
        reflect the new state.
        """
        logger.debug(
            f"Item data changed for {item.name}, updating properties UI."
        )
        self._update_ui()

    def _update_ui(self):
        """
        Updates the UI by querying all registered property providers and
        managing the visibility and content of their persistent widgets.
        """
        if not self.items:
            self._main_expander.set_sensitive(False)
            self._main_expander.set_title(_("Item Properties"))
            for provider, widgets in self._provider_widget_map:
                for widget in widgets:
                    widget.set_visible(False)
            for provider, widgets, expander in self._separate_groups:
                for widget in widgets:
                    widget.set_visible(False)
                expander.set_visible(False)
            return

        self._main_expander.set_sensitive(True)
        self._update_title(self.items[0])

        for provider, widgets in self._provider_widget_map:
            can_handle = provider.can_handle(self.items)
            for widget in widgets:
                widget.set_visible(can_handle)

            if can_handle:
                provider.update_widgets(self.editor, self.items)

        for provider, widgets, expander in self._separate_groups:
            can_handle = provider.can_handle(self.items)
            for widget in widgets:
                widget.set_visible(can_handle)
            expander.set_visible(can_handle)

            if can_handle:
                provider.update_widgets(self.editor, self.items)
                expander.set_subtitle(provider.group_subtitle)

    def _update_title(self, item: DocItem):
        """Sets the main title of the expander based on selection."""
        if len(self.items) > 1:
            self._main_expander.set_title(_("Multiple Items"))
        elif isinstance(item, StockItem):
            self._main_expander.set_title(_("Stock Properties"))
        elif isinstance(item, WorkPiece):
            self._main_expander.set_title(_("Workpiece Properties"))
        elif isinstance(item, Group):
            self._main_expander.set_title(_("Group Properties"))
        else:
            self._main_expander.set_title(_("Item Properties"))
