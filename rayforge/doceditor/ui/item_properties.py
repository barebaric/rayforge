import logging
from gi.repository import Gtk, Adw, Gio
from typing import Optional, Tuple, List, cast, TYPE_CHECKING
from ...context import get_context
from ...core.group import Group
from ...core.item import DocItem
from ...core.stock import StockItem
from ...core.workpiece import WorkPiece
from ...icons import get_icon
from ...shared.ui.expander import Expander
from ...shared.ui.adwfix import get_spinrow_float
from .image_metadata_dialog import ImageMetadataDialog

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


logger = logging.getLogger(__name__)
default_dim = 100, 100


class DocItemPropertiesWidget(Expander):
    def __init__(
        self,
        editor: "DocEditor",
        items: Optional[List[DocItem]] = None,
        *args,
        **kwargs,
    ):
        # Initialize the parent Expander widget
        super().__init__(*args, **kwargs)

        self.editor = editor
        self.items = items or []
        self._in_update = False

        # Set the title and default state on the Expander itself
        self.set_title(_("Item Properties"))
        self.set_expanded(True)  # Expanded by default

        # Create a ListBox to hold all the property rows. This replaces the
        # Adw.ExpanderRow's internal list.
        rows_container = Gtk.ListBox()
        rows_container.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(rows_container)

        # Source File Row
        self.source_file_row = Adw.ActionRow(
            title=_("Source File"),
            visible=False,  # Hidden by default
        )

        # Info button for metadata
        self.metadata_info_button = Gtk.Button()
        self.metadata_info_button.set_child(get_icon("info-symbolic"))
        self.metadata_info_button.set_valign(Gtk.Align.CENTER)
        self.metadata_info_button.set_tooltip_text(_("Show Image Metadata"))
        self.metadata_info_button.connect(
            "clicked", self._on_metadata_info_clicked
        )

        # Button to open file in file browser.
        self.source_file_row.add_suffix(self.metadata_info_button)
        self.open_source_button = Gtk.Button()
        self.open_source_button.set_child(get_icon("open-in-new-symbolic"))
        self.open_source_button.set_valign(Gtk.Align.CENTER)
        self.open_source_button.set_tooltip_text(_("Show in File Browser"))
        self.open_source_button.connect(
            "clicked", self._on_open_source_file_clicked
        )
        self.source_file_row.add_suffix(self.open_source_button)
        rows_container.append(self.source_file_row)

        # Vector count row
        self.vector_count_row = Adw.ActionRow(
            title=_("Vector Commands"),
            visible=False,
        )
        rows_container.append(self.vector_count_row)

        # X Position Entry
        self.x_row = Adw.SpinRow(
            title=_("X Position"),
            subtitle=_("Zero is on the left side"),
            adjustment=Gtk.Adjustment.new(0, -10000, 10000, 1.0, 1, 0),
        )
        self.x_row.set_digits(2)
        self.x_row.connect("notify::value", self._on_x_changed)
        rows_container.append(self.x_row)

        # Y Position Entry
        self.y_row = Adw.SpinRow(
            title=_("Y Position"),
            adjustment=Gtk.Adjustment.new(0, -10000, 10000, 1.0, 1, 0),
        )
        self.y_row.set_digits(2)
        self.y_row.connect("notify::value", self._on_y_changed)
        rows_container.append(self.y_row)

        # Fixed Ratio Switch
        self.fixed_ratio_switch = Adw.SwitchRow(
            title=_("Fixed Ratio"), active=True
        )
        self.fixed_ratio_switch.connect(
            "notify::active", self._on_fixed_ratio_toggled
        )
        rows_container.append(self.fixed_ratio_switch)

        # Width Entry
        self.width_row = Adw.SpinRow(
            title=_("Width"),
            adjustment=Gtk.Adjustment.new(10, 1, 10000, 1.0, 1, 0),
        )
        self.width_row.set_digits(2)
        self.width_row.connect("notify::value", self._on_width_changed)
        rows_container.append(self.width_row)

        # Height Entry
        self.height_row = Adw.SpinRow(
            title=_("Height"),
            adjustment=Gtk.Adjustment.new(10, 1, 10000, 1.0, 1, 0),
        )
        self.height_row.set_digits(2)
        self.height_row.connect("notify::value", self._on_height_changed)
        rows_container.append(self.height_row)

        # Angle Entry
        self.angle_row = Adw.SpinRow(
            title=_("Angle"),
            subtitle=_("Clockwise is positive"),
            adjustment=Gtk.Adjustment.new(0, -360, 360, 1, 10, 0),
            digits=2,
        )
        self.angle_row.connect("notify::value", self._on_angle_changed)
        rows_container.append(self.angle_row)

        # Shear Entry
        self.shear_row = Adw.SpinRow(
            title=_("Shear"),
            subtitle=_("Horizontal shear angle"),
            adjustment=Gtk.Adjustment.new(0, -85, 85, 1, 10, 0),
            digits=2,
        )
        self.shear_row.connect("notify::value", self._on_shear_changed)
        rows_container.append(self.shear_row)

        # Tabs Switch
        self.tabs_row = Adw.SwitchRow(
            title=_("Tabs"),
            visible=False,
        )
        self.tabs_row.connect("notify::active", self._on_tabs_enabled_toggled)
        rows_container.append(self.tabs_row)

        self.clear_tabs_button = Gtk.Button()
        self.clear_tabs_button.set_icon_name("edit-clear-symbolic")
        self.clear_tabs_button.set_valign(Gtk.Align.CENTER)
        self.clear_tabs_button.set_tooltip_text(_("Remove all tabs"))
        self.clear_tabs_button.connect("clicked", self._on_clear_tabs_clicked)
        self.tabs_row.add_suffix(self.clear_tabs_button)

        # Tab Width Entry
        self.tab_width_row = Adw.SpinRow(
            title=_("Tab Width"),
            subtitle=_(
                "Length along the path"
            ),  # Clarify what "width" means for a tab
            adjustment=Gtk.Adjustment.new(1.0, 0.1, 100.0, 0.1, 1.0, 0),
            digits=2,
            visible=False,  # Hidden by default
        )
        self.tab_width_row.connect("notify::value", self._on_tab_width_changed)
        rows_container.append(self.tab_width_row)

        # --- Reset Buttons ---
        def create_reset_button(tooltip_text, on_clicked):
            button = Gtk.Button.new_from_icon_name("edit-undo-symbolic")
            button.set_valign(Gtk.Align.CENTER)
            button.set_tooltip_text(tooltip_text)
            button.connect("clicked", on_clicked)
            return button

        self.reset_x_button = create_reset_button(
            _("Reset X position to 0"), self._on_reset_x_clicked
        )
        self.x_row.add_suffix(self.reset_x_button)

        self.reset_y_button = create_reset_button(
            _("Reset Y position to 0"), self._on_reset_y_clicked
        )
        self.y_row.add_suffix(self.reset_y_button)

        self.reset_width_button = create_reset_button(
            _("Reset to natural width"),
            lambda btn: self._on_reset_dimension_clicked(btn, "width"),
        )
        self.width_row.add_suffix(self.reset_width_button)

        self.reset_height_button = create_reset_button(
            _("Reset to natural height"),
            lambda btn: self._on_reset_dimension_clicked(btn, "height"),
        )
        self.height_row.add_suffix(self.reset_height_button)

        self.reset_aspect_button = create_reset_button(
            _("Reset to natural aspect ratio"), self._on_reset_aspect_clicked
        )
        self.fixed_ratio_switch.add_suffix(self.reset_aspect_button)

        self.reset_angle_button = create_reset_button(
            _("Reset angle to 0°"), self._on_reset_angle_clicked
        )
        self.angle_row.add_suffix(self.reset_angle_button)

        self.reset_shear_button = create_reset_button(
            _("Reset shear to 0°"), self._on_reset_shear_clicked
        )
        self.shear_row.add_suffix(self.reset_shear_button)

        self.reset_tab_width_button = create_reset_button(
            _("Reset tab width to default (1.0 mm)"),
            self._on_reset_tab_width_clicked,
        )
        self.tab_width_row.add_suffix(self.reset_tab_width_button)

        self.set_items(items)

    def _on_clear_tabs_clicked(self, button):
        if len(self.items) != 1 or not isinstance(self.items[0], WorkPiece):
            return
        workpiece = self.items[0]
        self.editor.tab.clear_tabs(workpiece)

    def _on_tabs_enabled_toggled(self, switch, GParamSpec):
        if self._in_update or not self.items:
            return

        # Should only affect a single selected workpiece
        if len(self.items) != 1 or not isinstance(self.items[0], WorkPiece):
            return

        workpiece = cast(WorkPiece, self.items[0])
        new_value = switch.get_active()
        self.editor.tab.set_workpiece_tabs_enabled(workpiece, new_value)

    def _on_tab_width_changed(self, spin_row, GParamSpec):
        if self._in_update or not self.items:
            return

        if len(self.items) != 1 or not isinstance(self.items[0], WorkPiece):
            return

        workpiece = cast(WorkPiece, self.items[0])
        new_width = get_spinrow_float(self.tab_width_row)
        if new_width is None or new_width <= 0:  # Ensure valid width
            return

        self.editor.tab.set_workpiece_tab_width(workpiece, new_width)

    def _on_reset_tab_width_clicked(self, button):
        if len(self.items) != 1 or not isinstance(self.items[0], WorkPiece):
            return

        workpiece = cast(WorkPiece, self.items[0])
        default_width = (
            1.0  # Default width, matching the spinrow's initial value
        )
        self.editor.tab.set_workpiece_tab_width(workpiece, default_width)

    def _calculate_new_size_with_ratio(
        self, item: DocItem, value: float, changed_dim: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculates new width and height maintaining aspect ratio."""
        # This now works for both WorkPiece and StockItem via duck-typing.
        aspect_ratio = None
        if isinstance(item, (WorkPiece, StockItem, Group)):
            aspect_ratio = item.get_current_aspect_ratio()

        if not aspect_ratio:
            return None, None

        width_min = self.width_row.get_adjustment().get_lower()
        height_min = self.height_row.get_adjustment().get_lower()

        if changed_dim == "width":
            new_width = value
            new_height = new_width / aspect_ratio
            if new_height < height_min:
                new_height = height_min
                new_width = new_height * aspect_ratio
        else:  # changed_dim == 'height'
            new_height = value
            new_width = new_height * aspect_ratio
            if new_width < width_min:
                new_width = width_min
                new_height = new_width / aspect_ratio

        return new_width, new_height

    def _on_width_changed(self, spin_row, GParamSpec):
        logger.debug(f"Width changed to {spin_row.get_value()}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_width_from_ui = get_spinrow_float(self.width_row)
            if new_width_from_ui is None:
                return

            # Use the first item to update the UI height if ratio is fixed
            if self.fixed_ratio_switch.get_active():
                first_item = self.items[0]
                w, h = self._calculate_new_size_with_ratio(
                    first_item, new_width_from_ui, "width"
                )
                if w is not None and h is not None:
                    self.height_row.set_value(h)
                    self.width_row.set_value(w)

            self.editor.transform.set_size(
                items=self.items,
                width=get_spinrow_float(self.width_row),
                height=get_spinrow_float(self.height_row),
                fixed_ratio=self.fixed_ratio_switch.get_active(),
            )
        finally:
            self._in_update = False

    def _on_height_changed(self, spin_row, GParamSpec):
        logger.debug(f"Height changed to {spin_row.get_value()}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_height_from_ui = get_spinrow_float(self.height_row)
            if new_height_from_ui is None:
                return

            # Use the first item to update UI width if ratio is fixed
            if self.fixed_ratio_switch.get_active():
                first_item = self.items[0]
                w, h = self._calculate_new_size_with_ratio(
                    first_item, new_height_from_ui, "height"
                )
                if w is not None and h is not None:
                    self.width_row.set_value(w)
                    self.height_row.set_value(h)

            self.editor.transform.set_size(
                items=self.items,
                width=get_spinrow_float(self.width_row),
                height=get_spinrow_float(self.height_row),
                fixed_ratio=self.fixed_ratio_switch.get_active(),
            )
        finally:
            self._in_update = False

    def _on_x_changed(self, spin_row, GParamSpec):
        logger.debug(f"X position changed to {spin_row.get_value()}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_x_machine = get_spinrow_float(self.x_row)
            if new_x_machine is None:
                return

            # Get current Y from the UI widget
            current_y_machine = self.y_row.get_value()
            self.editor.transform.set_position(
                self.items, new_x_machine, current_y_machine
            )
        finally:
            self._in_update = False

    def _on_y_changed(self, spin_row, GParamSpec):
        logger.debug(f"Y position changed to {spin_row.get_value()}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_y_machine = get_spinrow_float(self.y_row)
            if new_y_machine is None:
                return

            # Get current X from the UI widget
            current_x_machine = self.x_row.get_value()
            self.editor.transform.set_position(
                self.items, current_x_machine, new_y_machine
            )
        finally:
            self._in_update = False

    def _on_angle_changed(self, spin_row, GParamSpec):
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_angle_from_ui = spin_row.get_value()
            new_angle = -new_angle_from_ui

            self.editor.transform.set_angle(self.items, new_angle)
        finally:
            self._in_update = False

    def _on_shear_changed(self, spin_row, GParamSpec):
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_shear_from_ui = spin_row.get_value()

            self.editor.transform.set_shear(self.items, new_shear_from_ui)
        finally:
            self._in_update = False

    def _on_fixed_ratio_toggled(self, switch_row, GParamSpec):
        logger.debug(f"Fixed ratio toggled: {switch_row.get_active()}")
        # Check if the primary selected item is a workpiece or stock item
        is_ratio_lockable = self.items and isinstance(
            self.items[0], (WorkPiece, StockItem, Group)
        )
        if not is_ratio_lockable:
            # For groups or multi-select, lock-ratio doesn't have a clear
            # definition of 'natural aspect', so we disable it.
            switch_row.set_sensitive(False)
        else:
            switch_row.set_sensitive(True)

    def _on_open_source_file_clicked(self, button):
        if len(self.items) != 1 or not isinstance(self.items[0], WorkPiece):
            return

        workpiece = cast(WorkPiece, self.items[0])
        file_path = workpiece.source_file

        if file_path and file_path.is_file():
            try:
                gio_file = Gio.File.new_for_path(str(file_path.resolve()))
                launcher = Gtk.FileLauncher.new(gio_file)
                window = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
                launcher.open_containing_folder(window, None, None)
            except Exception as e:
                logger.error(f"Failed to show file in browser: {e}")

    def _on_metadata_info_clicked(self, button):
        """Handles the metadata info button click."""
        if len(self.items) != 1 or not isinstance(self.items[0], WorkPiece):
            return

        workpiece = cast(WorkPiece, self.items[0])
        source = workpiece.source

        if not source or not source.metadata:
            return

        # Create and show the metadata dialog
        root = self.get_root()
        dialog = ImageMetadataDialog(
            parent=root if isinstance(root, Gtk.Window) else None
        )
        dialog.set_metadata(source)
        dialog.present()

    def _on_reset_aspect_clicked(self, button):
        if not self.items:
            return

        items_to_resize = []
        for item in self.items:
            if not isinstance(item, (WorkPiece, StockItem, Group)):
                continue
            current_size = item.size
            current_width = current_size[0]

            default_aspect = None
            if isinstance(item, (WorkPiece, StockItem)):
                default_aspect = item.get_natural_aspect_ratio()
            elif item.natural_size:
                nw, nh = item.natural_size
                default_aspect = nw / nh if nh > 0 else None

            if not default_aspect or default_aspect == 0:
                continue
            new_height = current_width / default_aspect
            new_size = (current_width, new_height)
            if new_size != current_size:
                items_to_resize.append(item)

        if items_to_resize:
            # This command needs to be more complex to handle individual
            # aspect ratios. For now, we use the first item's aspect ratio.
            # This is a limitation of the new command structure.
            # A better approach would be to pass a list of (uid, w, h) tuples.
            first_item = items_to_resize[0]
            if isinstance(first_item, (WorkPiece, StockItem, Group)):
                current_width = first_item.size[0]

                default_aspect = None
                if isinstance(first_item, (WorkPiece, StockItem)):
                    default_aspect = first_item.get_natural_aspect_ratio()
                elif first_item.natural_size:
                    nw, nh = first_item.natural_size
                    default_aspect = nw / nh if nh > 0 else None

                if default_aspect and default_aspect != 0:
                    new_height = current_width / default_aspect
                    self.editor.transform.set_size(
                        items=items_to_resize,
                        width=current_width,
                        height=new_height,
                        fixed_ratio=False,  # We are setting the exact size
                    )

    def _on_reset_dimension_clicked(self, button, dimension_to_reset: str):
        if not self.items:
            return

        sizes_to_set = []
        items_to_resize = []
        for item in self.items:
            if not isinstance(item, (WorkPiece, StockItem, Group)):
                continue

            natural_width, natural_height = item.natural_size
            current_width, current_height = item.size

            new_width = current_width
            new_height = current_height

            if dimension_to_reset == "width":
                new_width = natural_width
                if self.fixed_ratio_switch.get_active():
                    aspect = None
                    if isinstance(item, (WorkPiece, StockItem)):
                        aspect = item.get_natural_aspect_ratio()
                    elif item.natural_size:
                        nw, nh = item.natural_size
                        aspect = nw / nh if nh > 0 else None

                    if aspect and new_width > 1e-9:
                        new_height = new_width / aspect
            else:  # dimension_to_reset == "height"
                new_height = natural_height
                if self.fixed_ratio_switch.get_active():
                    aspect = None
                    if isinstance(item, (WorkPiece, StockItem)):
                        aspect = item.get_natural_aspect_ratio()
                    elif item.natural_size:
                        nw, nh = item.natural_size
                        aspect = nw / nh if nh > 0 else None

                    if aspect and new_height > 1e-9:
                        new_width = new_height * aspect

            new_size = (new_width, new_height)
            if new_size != item.size:
                items_to_resize.append(item)
                sizes_to_set.append(new_size)

        if items_to_resize:
            self.editor.transform.set_size(
                items=items_to_resize,
                sizes=sizes_to_set,
            )

    def _on_reset_angle_clicked(self, button):
        if not self.items:
            return
        items_to_reset = [item for item in self.items if item.angle != 0.0]
        if items_to_reset:
            self.editor.transform.set_angle(items_to_reset, 0.0)

    def _on_reset_shear_clicked(self, button):
        if not self.items:
            return
        items_to_reset = [item for item in self.items if item.shear != 0.0]
        if items_to_reset:
            self.editor.transform.set_shear(items_to_reset, 0.0)

    def _on_reset_x_clicked(self, button):
        if not self.items:
            return
        items_to_reset = [
            item for item in self.items if abs(item.pos[0] - 0.0) >= 1e-9
        ]
        if items_to_reset:
            # Get current Y from the UI widget
            current_y_machine = self.y_row.get_value()
            self.editor.transform.set_position(
                items_to_reset, 0.0, current_y_machine
            )

    def _on_reset_y_clicked(self, button):
        if not self.items:
            return
        machine = get_context().machine
        bounds = machine.dimensions if machine else default_dim
        y_axis_down = machine.y_axis_down if machine else False

        items_to_reset = []
        target_y_machine = 0.0
        if y_axis_down:
            machine_height = bounds[1]
            # For multi-select, we can't have different target Ys.
            # This is a limitation of simplifying to a single command.
            first_item_size = self.items[0].size
            target_y_machine = machine_height - first_item_size[1]

        for item in self.items:
            pos_world = item.pos
            size_world = item.size
            target_y_world = 0.0

            if y_axis_down:
                machine_height = bounds[1]
                target_y_world = machine_height - size_world[1]

            if abs(pos_world[1] - target_y_world) >= 1e-9:
                items_to_reset.append(item)

        if items_to_reset:
            # Get current X from the UI widget
            current_x_machine = self.x_row.get_value()
            self.editor.transform.set_position(
                items_to_reset, current_x_machine, target_y_machine
            )

    def _on_item_data_changed(self, item):
        """
        Handles data changes from the DocItem model. This will now be
        triggered for both size and transform changes.
        """
        if self._in_update:
            return
        logger.debug(f"Item data changed: {item.name}")
        self._update_ui_from_items()

    def set_items(self, items: Optional[List[DocItem]]):
        for item in self.items:
            item.updated.disconnect(self._on_item_data_changed)
            item.transform_changed.disconnect(self._on_item_data_changed)

        self.items = items or []

        count = len(self.items)
        if count == 1:
            self.set_subtitle(_("1 item selected"))
        else:
            self.set_subtitle(_(f"{count} items selected"))

        for item in self.items:
            item.updated.connect(self._on_item_data_changed)
            item.transform_changed.connect(self._on_item_data_changed)

        self._update_ui_from_items()

    def _update_title(self, item: DocItem):
        if len(self.items) > 1:
            self.set_title(_("Multiple Items"))
        elif isinstance(item, StockItem):
            self.set_title(_("Stock Properties"))
        elif isinstance(item, WorkPiece):
            self.set_title(_("Workpiece Properties"))
        elif isinstance(item, Group):
            self.set_title(_("Group Properties"))
        else:
            self.set_title(_("Item Properties"))

    def _update_main_properties(self, item: DocItem):
        machine = get_context().machine
        bounds = machine.dimensions if machine else default_dim
        y_axis_down = machine.y_axis_down if machine else False

        size_world = item.size
        pos_world = item.pos
        angle_local = item.angle
        shear_local = item.shear

        if y_axis_down:
            self.y_row.set_subtitle(_("Zero is at the top"))
            machine_height = bounds[1]
            pos_machine_x = pos_world[0]
            pos_machine_y = machine_height - pos_world[1] - size_world[1]
        else:
            self.y_row.set_subtitle(_("Zero is at the bottom"))
            pos_machine_x, pos_machine_y = pos_world

        self.width_row.set_value(size_world[0])
        self.height_row.set_value(size_world[1])
        self.x_row.set_value(pos_machine_x)
        self.y_row.set_value(pos_machine_y)
        self.angle_row.set_value(-angle_local)
        self.shear_row.set_value(shear_local)

    def _update_row_visibility_and_details(self, item: DocItem):
        is_single_workpiece = len(self.items) == 1 and isinstance(
            item, WorkPiece
        )
        is_single_stockitem = len(self.items) == 1 and isinstance(
            item, StockItem
        )
        is_single_group = len(self.items) == 1 and isinstance(item, Group)
        is_single_item_with_size = (
            is_single_workpiece or is_single_stockitem or is_single_group
        )

        self.source_file_row.set_visible(is_single_workpiece)
        self.fixed_ratio_switch.set_sensitive(
            is_single_item_with_size or is_single_group
        )
        self.reset_width_button.set_sensitive(is_single_item_with_size)
        self.reset_height_button.set_sensitive(is_single_item_with_size)
        self.reset_aspect_button.set_sensitive(is_single_item_with_size)
        self.shear_row.set_visible(not isinstance(item, Group))

        if is_single_item_with_size:
            natural_width, natural_height = None, None

            if isinstance(item, (WorkPiece, StockItem)):
                machine = get_context().machine
                bounds = machine.dimensions if machine else default_dim
                natural_width, natural_height = item.get_default_size(*bounds)
            elif item.natural_size:
                natural_width, natural_height = item.natural_size

            if natural_width is not None:
                self.width_row.set_subtitle(
                    _("Natural: {val:.2f}").format(val=natural_width)
                )
                self.height_row.set_subtitle(
                    _("Natural: {val:.2f}").format(val=natural_height)
                )
            else:
                self.width_row.set_subtitle("")
                self.height_row.set_subtitle("")
        else:
            self.width_row.set_subtitle("")
            self.height_row.set_subtitle("")

        if is_single_workpiece:
            self._update_workpiece_specific_rows(cast(WorkPiece, item))
        else:
            self.tabs_row.set_visible(False)
            self.tab_width_row.set_visible(False)
            self.vector_count_row.set_visible(False)

    def _update_workpiece_specific_rows(self, workpiece: WorkPiece):
        is_debug_and_has_vectors = (
            logging.getLogger().getEffectiveLevel() == logging.DEBUG
            and workpiece.boundaries is not None
        )
        self.vector_count_row.set_visible(is_debug_and_has_vectors)
        if is_debug_and_has_vectors:
            vectors = len(workpiece.boundaries) if workpiece.boundaries else 0
            self.vector_count_row.set_subtitle(f"{vectors} commands")

        self._update_source_file_row(workpiece)
        self._update_tabs_rows(workpiece)

    def _update_source_file_row(self, workpiece: WorkPiece):
        file_path = workpiece.source_file
        if file_path:
            if file_path.is_file():
                self.source_file_row.set_subtitle(file_path.name)
                self.open_source_button.set_sensitive(True)
                # Enable metadata button if source has metadata
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

    def _update_tabs_rows(self, workpiece: WorkPiece):
        can_have_tabs = workpiece.boundaries is not None
        self.tabs_row.set_visible(can_have_tabs)
        self.tab_width_row.set_visible(
            can_have_tabs and workpiece.tabs_enabled
        )

        if not can_have_tabs:
            return

        self.tabs_row.set_active(workpiece.tabs_enabled)
        self.clear_tabs_button.set_sensitive(bool(workpiece.tabs))
        self.tabs_row.set_subtitle(
            _("{num_tabs} tabs").format(num_tabs=len(workpiece.tabs))
        )

        if workpiece.tabs_enabled:
            self._update_tab_width_row_for_enabled_tabs(workpiece)
        else:
            self.tab_width_row.set_value(1.0)
            self.tab_width_row.set_subtitle(_("Length along the path"))
            self.tab_width_row.set_sensitive(False)
            self.reset_tab_width_button.set_sensitive(False)

    def _update_tab_width_row_for_enabled_tabs(self, workpiece: WorkPiece):
        if workpiece.tabs:
            first_tab_width = workpiece.tabs[0].width
            self.tab_width_row.set_value(first_tab_width)
            if not all(t.width == first_tab_width for t in workpiece.tabs):
                self.tab_width_row.set_subtitle(_("Mixed values"))
            else:
                self.tab_width_row.set_subtitle(_("Length along the path"))
            self.tab_width_row.set_sensitive(True)
            self.reset_tab_width_button.set_sensitive(True)
        else:  # Tabs enabled, but no tabs present
            self.tab_width_row.set_value(1.0)
            self.tab_width_row.set_subtitle(_("Length along the path"))
            # Cannot change width if no tabs to modify
            self.tab_width_row.set_sensitive(False)
            self.reset_tab_width_button.set_sensitive(False)

    def _update_ui_from_items(self):
        logger.debug(f"Updating UI for items: {self.items}")
        if not self.items:
            self.set_sensitive(False)
            return

        self.set_sensitive(True)
        item = self.items[0]

        self._in_update = True
        try:
            self._update_title(item)
            self._update_main_properties(item)
            self._update_row_visibility_and_details(item)
        finally:
            self._in_update = False
