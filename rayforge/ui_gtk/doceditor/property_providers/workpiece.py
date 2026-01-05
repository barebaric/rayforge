import logging
from gi.repository import Gtk, Adw, Gio
from typing import List, cast, TYPE_CHECKING
from ....core.item import DocItem
from ....core.workpiece import WorkPiece
from ...icons import get_icon
from ...shared.adwfix import get_spinrow_float
from ..image_metadata_dialog import ImageMetadataDialog
from .base import PropertyProvider

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class WorkpieceInfoProvider(PropertyProvider):
    """Provides UI for Workpiece-specific info (source file, metadata)."""

    def can_handle(self, items: List[DocItem]) -> bool:
        return len(items) == 1 and isinstance(items[0], WorkPiece)

    def create_widgets(self) -> List[Gtk.Widget]:
        """Creates the widgets for workpiece info properties."""
        logger.debug("Creating workpiece info property widgets.")
        # Source File Row
        self.source_file_row = Adw.ActionRow(title=_("Source File"))
        self.metadata_info_button = Gtk.Button(
            child=get_icon("info-symbolic"),
            valign=Gtk.Align.CENTER,
            tooltip_text=_("Show Image Metadata"),
        )
        self.metadata_info_button.connect(
            "clicked", self._on_metadata_info_clicked
        )
        self.source_file_row.add_suffix(self.metadata_info_button)

        self.open_source_button = Gtk.Button(
            child=get_icon("open-in-new-symbolic"),
            valign=Gtk.Align.CENTER,
            tooltip_text=_("Show in File Browser"),
        )
        self.open_source_button.connect(
            "clicked", self._on_open_source_file_clicked
        )
        self.source_file_row.add_suffix(self.open_source_button)

        # Vector count row
        self.vector_count_row = Adw.ActionRow(title=_("Vector Commands"))

        return [self.source_file_row, self.vector_count_row]

    def update_widgets(self, editor: "DocEditor", items: List[DocItem]):
        """Updates the workpiece info widgets with new data."""
        logger.debug(
            f"Updating workpiece info widgets for {len(items)} items."
        )
        self.editor = editor
        self.items = items
        workpiece = cast(WorkPiece, self.items[0])

        self._update_source_file_row(workpiece)

        is_debug_and_has_vectors = (
            logging.getLogger().getEffectiveLevel() == logging.DEBUG
            and workpiece.boundaries is not None
        )
        self.vector_count_row.set_visible(is_debug_and_has_vectors)
        if is_debug_and_has_vectors:
            vectors = len(workpiece.boundaries) if workpiece.boundaries else 0
            self.vector_count_row.set_subtitle(f"{vectors} commands")

    def _update_source_file_row(self, workpiece: WorkPiece):
        file_path = workpiece.source_file
        if file_path:
            if file_path.is_file():
                self.source_file_row.set_subtitle(file_path.name)
                self.open_source_button.set_sensitive(True)
                source = workpiece.source
                has_metadata = bool(
                    source and source.metadata and len(source.metadata) > 0
                )
                self.metadata_info_button.set_sensitive(has_metadata)
            else:
                self.source_file_row.set_subtitle(
                    _("{name} (not found)").format(name=file_path.name)
                )
                self.open_source_button.set_sensitive(False)
                self.metadata_info_button.set_sensitive(False)
        else:
            self.source_file_row.set_subtitle(_("(No source file)"))
            self.open_source_button.set_sensitive(False)
            self.metadata_info_button.set_sensitive(False)

    def _on_open_source_file_clicked(self, button):
        workpiece = cast(WorkPiece, self.items[0])
        file_path = workpiece.source_file
        if file_path and file_path.is_file():
            try:
                gio_file = Gio.File.new_for_path(str(file_path.resolve()))
                launcher = Gtk.FileLauncher.new(gio_file)
                window = cast(
                    Gtk.Window, self.source_file_row.get_ancestor(Gtk.Window)
                )
                launcher.open_containing_folder(window, None, None)
            except Exception as e:
                logger.error(f"Failed to show file in browser: {e}")

    def _on_metadata_info_clicked(self, button):
        workpiece = cast(WorkPiece, self.items[0])
        source = workpiece.source
        if not source or not source.metadata:
            return

        root = self.source_file_row.get_root()
        dialog = ImageMetadataDialog(
            parent=root if isinstance(root, Gtk.Window) else None
        )
        dialog.set_metadata(source)
        dialog.present()


class TabsPropertyProvider(PropertyProvider):
    """Provides UI for managing tabs on a Workpiece."""

    def can_handle(self, items: List[DocItem]) -> bool:
        return (
            len(items) == 1
            and isinstance(items[0], WorkPiece)
            and items[0].boundaries is not None
        )

    def create_widgets(self) -> List[Gtk.Widget]:
        """Creates the widgets for tab properties."""
        logger.debug("Creating tabs property widgets.")
        self._rows = []

        # Tabs Switch
        self.tabs_row = Adw.SwitchRow(title=_("Tabs"))
        self.tabs_row.connect("notify::active", self._on_tabs_enabled_toggled)

        self.clear_tabs_button = Gtk.Button(
            child=get_icon("clear-symbolic"),
            valign=Gtk.Align.CENTER,
            tooltip_text=_("Remove all tabs"),
        )
        self.clear_tabs_button.connect("clicked", self._on_clear_tabs_clicked)
        self.tabs_row.add_suffix(self.clear_tabs_button)
        self._rows.append(self.tabs_row)

        # Tab Width Entry
        self.tab_width_row = Adw.SpinRow(
            title=_("Tab Width"),
            subtitle=_("Length along the path"),
            adjustment=Gtk.Adjustment.new(1.0, 0.1, 100.0, 0.1, 1.0, 0),
            digits=2,
        )
        self.tab_width_row.connect("notify::value", self._on_tab_width_changed)
        self.reset_tab_width_button = Gtk.Button(
            child=get_icon("undo-symbolic")
        )
        self.reset_tab_width_button.set_valign(Gtk.Align.CENTER)
        self.reset_tab_width_button.set_tooltip_text(
            _("Reset tab width to default (1.0 mm)")
        )
        self.reset_tab_width_button.connect(
            "clicked", self._on_reset_tab_width_clicked
        )
        self.tab_width_row.add_suffix(self.reset_tab_width_button)
        self._rows.append(self.tab_width_row)

        return self._rows

    def update_widgets(self, editor: "DocEditor", items: List[DocItem]):
        """Updates the tabs widgets with new data."""
        logger.debug(f"Updating tabs property widgets for {len(items)} items.")
        self.editor = editor
        self.items = items
        workpiece = cast(WorkPiece, self.items[0])
        self._update_tabs_rows(workpiece)

    def _update_tabs_rows(self, workpiece: WorkPiece):
        self._in_update = True
        try:
            self.tabs_row.set_active(workpiece.tabs_enabled)
        finally:
            self._in_update = False

        self.tab_width_row.set_visible(workpiece.tabs_enabled)
        self.clear_tabs_button.set_sensitive(bool(workpiece.tabs))
        self.tabs_row.set_subtitle(
            _("{num_tabs} tabs").format(num_tabs=len(workpiece.tabs))
        )

        if workpiece.tabs_enabled:
            if workpiece.tabs:
                first_tab_width = workpiece.tabs[0].width
                self.tab_width_row.set_value(first_tab_width)
                if not all(t.width == first_tab_width for t in workpiece.tabs):
                    self.tab_width_row.set_subtitle(_("Mixed values"))
                else:
                    self.tab_width_row.set_subtitle(_("Length along the path"))
                self.tab_width_row.set_sensitive(True)
                self.reset_tab_width_button.set_sensitive(True)
            else:
                self.tab_width_row.set_value(1.0)
                self.tab_width_row.set_subtitle(_("Length along the path"))
                self.tab_width_row.set_sensitive(False)
                self.reset_tab_width_button.set_sensitive(False)
        else:
            self.tab_width_row.set_value(1.0)
            self.tab_width_row.set_subtitle(_("Length along the path"))
            self.tab_width_row.set_sensitive(False)
            self.reset_tab_width_button.set_sensitive(False)

    def _on_clear_tabs_clicked(self, button):
        workpiece = cast(WorkPiece, self.items[0])
        self.editor.tab.clear_tabs(workpiece)

    def _on_tabs_enabled_toggled(self, switch, GParamSpec):
        logger.debug(
            f"_on_tabs_enabled_toggled called. _in_update={self._in_update}"
        )
        if self._in_update:
            return
        workpiece = cast(WorkPiece, self.items[0])
        new_value = switch.get_active()
        self.editor.tab.set_workpiece_tabs_enabled(workpiece, new_value)

    def _on_tab_width_changed(self, spin_row, GParamSpec):
        logger.debug(
            f"_on_tab_width_changed called. _in_update={self._in_update}"
        )
        if self._in_update:
            return
        workpiece = cast(WorkPiece, self.items[0])
        new_width = get_spinrow_float(self.tab_width_row)
        if new_width is None or new_width <= 0:
            return
        self.editor.tab.set_workpiece_tab_width(workpiece, new_width)

    def _on_reset_tab_width_clicked(self, button):
        workpiece = cast(WorkPiece, self.items[0])
        self.editor.tab.set_workpiece_tab_width(workpiece, 1.0)
