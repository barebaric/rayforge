import logging
from gi.repository import Gtk, Adw
from typing import Optional, Tuple, List, TYPE_CHECKING
from ....context import get_context
from ....core.group import Group
from ....core.item import DocItem
from ....core.stock import StockItem
from ....core.workpiece import WorkPiece
from ...shared.adwfix import get_spinrow_float
from .base import PropertyProvider

if TYPE_CHECKING:
    from ....doceditor.editor import DocEditor

default_dim = 100, 100
logger = logging.getLogger(__name__)


class TransformPropertyProvider(PropertyProvider):
    """Provides UI for common transformation properties (pos, size, angle)."""

    def can_handle(self, items: List[DocItem]) -> bool:
        return bool(items)

    def create_widgets(self) -> List[Gtk.Widget]:
        """Creates the widgets for transform properties once."""
        logger.debug("Creating transform property widgets.")
        self._rows = []
        self._create_position_rows()
        self._create_size_rows()
        self._create_angle_shear_rows()
        return self._rows

    def update_widgets(self, editor: "DocEditor", items: List[DocItem]):
        """Updates the transform widgets with data from the selected items."""
        logger.debug(f"Updating transform widgets for {len(items)} items.")
        self.editor = editor
        self.items = items

        item = self.items[0]
        machine = get_context().machine
        bounds = machine.dimensions if machine else default_dim
        x_axis_right = machine.x_axis_right if machine else False
        y_axis_down = machine.y_axis_down if machine else False

        size_world = item.size
        pos_world = item.pos
        angle_local = item.angle
        shear_local = item.shear

        # Calculate X position in machine coordinates
        if x_axis_right:
            self.x_row.set_subtitle(_("Zero is on the right"))
            machine_width = bounds[0]
            pos_machine_x = machine_width - pos_world[0] - size_world[0]
        else:
            self.x_row.set_subtitle(_("Zero is on the left"))
            pos_machine_x = pos_world[0]

        # Calculate Y position in machine coordinates
        if y_axis_down:
            self.y_row.set_subtitle(_("Zero is at the top"))
            machine_height = bounds[1]
            pos_machine_y = machine_height - pos_world[1] - size_world[1]
        else:
            self.y_row.set_subtitle(_("Zero is at the bottom"))
            pos_machine_y = pos_world[1]

        # Use a safe re-entrant pattern for updating widgets
        was_in_update = getattr(self, "_in_update", False)
        self._in_update = True
        try:
            # Get digits for rounding to prevent float precision issues
            width_digits = self.width_row.get_digits()
            height_digits = self.height_row.get_digits()
            x_digits = self.x_row.get_digits()
            y_digits = self.y_row.get_digits()
            angle_digits = self.angle_row.get_digits()
            shear_digits = self.shear_row.get_digits()

            # Round values before comparing and setting
            width_rounded = round(size_world[0], width_digits)
            height_rounded = round(size_world[1], height_digits)
            x_rounded = round(pos_machine_x, x_digits)
            y_rounded = round(pos_machine_y, y_digits)
            angle_rounded = round(-angle_local, angle_digits)
            shear_rounded = round(shear_local, shear_digits)

            # Check before setting to avoid signal emission loop
            if abs(self.width_row.get_value() - width_rounded) > 1e-9:
                logger.debug(f"Setting width UI to {width_rounded}")
                self.width_row.set_value(width_rounded)
            if abs(self.height_row.get_value() - height_rounded) > 1e-9:
                logger.debug(f"Setting height UI to {height_rounded}")
                self.height_row.set_value(height_rounded)

            if abs(self.x_row.get_value() - x_rounded) > 1e-9:
                logger.debug(f"Setting X UI to {x_rounded}")
                self.x_row.set_value(x_rounded)

            if abs(self.y_row.get_value() - y_rounded) > 1e-9:
                logger.debug(f"Setting Y UI to {y_rounded}")
                self.y_row.set_value(y_rounded)

            if abs(self.angle_row.get_value() - angle_rounded) > 1e-9:
                logger.debug(f"Setting angle UI to {angle_rounded}")
                self.angle_row.set_value(angle_rounded)
            if abs(self.shear_row.get_value() - shear_rounded) > 1e-9:
                logger.debug(f"Setting shear UI to {shear_rounded}")
                self.shear_row.set_value(shear_rounded)
        finally:
            self._in_update = was_in_update

        self._update_row_visibility_and_details()

    def _create_position_rows(self):
        # X Position Entry
        self.x_row = Adw.SpinRow(
            title=_("X Position"),
            subtitle=_("Zero is on the left side"),
            adjustment=Gtk.Adjustment.new(0, -10000, 10000, 1.0, 1, 0),
        )
        self.x_row.set_digits(2)
        self.x_row.connect("notify::value", self._on_x_changed)

        # X Reset Button
        self.reset_x_button = self._create_reset_button(
            _("Reset X position to 0"), self._on_reset_x_clicked
        )
        self.x_row.add_suffix(self.reset_x_button)

        # Y Position Entry
        self.y_row = Adw.SpinRow(
            title=_("Y Position"),
            adjustment=Gtk.Adjustment.new(0, -10000, 10000, 1.0, 1, 0),
        )
        self.y_row.set_digits(2)
        self.y_row.connect("notify::value", self._on_y_changed)

        self.reset_y_button = self._create_reset_button(
            _("Reset Y position to 0"), self._on_reset_y_clicked
        )
        self.y_row.add_suffix(self.reset_y_button)

        self._rows.extend([self.x_row, self.y_row])

    def _create_size_rows(self):
        # Fixed Ratio Switch
        self.fixed_ratio_switch = Adw.SwitchRow(
            title=_("Fixed Ratio"), active=True
        )
        self.fixed_ratio_switch.connect(
            "notify::active", self._on_fixed_ratio_toggled
        )

        # Width Entry
        self.width_row = Adw.SpinRow(
            title=_("Width"),
            adjustment=Gtk.Adjustment.new(10, 1, 10000, 1.0, 1, 0),
        )
        self.width_row.set_digits(2)
        self.width_row.connect("notify::value", self._on_width_changed)

        # Height Entry
        self.height_row = Adw.SpinRow(
            title=_("Height"),
            adjustment=Gtk.Adjustment.new(10, 1, 10000, 1.0, 1, 0),
        )
        self.height_row.set_digits(2)
        self.height_row.connect("notify::value", self._on_height_changed)

        # Reset Buttons
        self.reset_width_button = self._create_reset_button(
            _("Reset to natural width"),
            lambda btn: self._on_reset_dimension_clicked(btn, "width"),
        )
        self.width_row.add_suffix(self.reset_width_button)

        self.reset_height_button = self._create_reset_button(
            _("Reset to natural height"),
            lambda btn: self._on_reset_dimension_clicked(btn, "height"),
        )
        self.height_row.add_suffix(self.reset_height_button)

        self.reset_aspect_button = self._create_reset_button(
            _("Reset to natural aspect ratio"), self._on_reset_aspect_clicked
        )
        self.fixed_ratio_switch.add_suffix(self.reset_aspect_button)

        self._rows.extend(
            [self.fixed_ratio_switch, self.width_row, self.height_row]
        )

    def _create_angle_shear_rows(self):
        # Angle Entry
        self.angle_row = Adw.SpinRow(
            title=_("Angle"),
            subtitle=_("Clockwise is positive"),
            adjustment=Gtk.Adjustment.new(0, -360, 360, 1, 10, 0),
            digits=2,
        )
        self.angle_row.connect("notify::value", self._on_angle_changed)

        # Shear Entry
        self.shear_row = Adw.SpinRow(
            title=_("Shear"),
            subtitle=_("Horizontal shear angle"),
            adjustment=Gtk.Adjustment.new(0, -85, 85, 1, 10, 0),
            digits=2,
        )
        self.shear_row.connect("notify::value", self._on_shear_changed)

        # Reset Buttons
        self.reset_angle_button = self._create_reset_button(
            _("Reset angle to 0°"), self._on_reset_angle_clicked
        )
        self.angle_row.add_suffix(self.reset_angle_button)

        self.reset_shear_button = self._create_reset_button(
            _("Reset shear to 0°"), self._on_reset_shear_clicked
        )
        self.shear_row.add_suffix(self.reset_shear_button)

        self._rows.extend([self.angle_row, self.shear_row])

    def _update_row_visibility_and_details(self):
        item = self.items[0]
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

    def _create_reset_button(self, tooltip_text, on_clicked):
        button = Gtk.Button.new_from_icon_name("edit-undo-symbolic")
        button.set_valign(Gtk.Align.CENTER)
        button.set_tooltip_text(tooltip_text)
        button.connect("clicked", on_clicked)
        return button

    def _calculate_new_size_with_ratio(
        self, item: DocItem, value: float, changed_dim: str
    ) -> Tuple[Optional[float], Optional[float]]:
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
        logger.debug(f"_on_width_changed called. _in_update={self._in_update}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_width_from_ui = get_spinrow_float(self.width_row)
            if new_width_from_ui is None:
                logger.debug("Width change ignored, no value from UI.")
                return

            logger.debug(f"Handling width change to {new_width_from_ui}")

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
            logger.debug("_on_width_changed finished.")

    def _on_height_changed(self, spin_row, GParamSpec):
        logger.debug(
            f"_on_height_changed called. _in_update={self._in_update}"
        )
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_height_from_ui = get_spinrow_float(self.height_row)
            if new_height_from_ui is None:
                logger.debug("Height change ignored, no value from UI.")
                return

            logger.debug(f"Handling height change to {new_height_from_ui}")

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
            logger.debug("_on_height_changed finished.")

    def _on_x_changed(self, spin_row, GParamSpec):
        logger.debug(f"_on_x_changed called. _in_update={self._in_update}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_x_machine = get_spinrow_float(self.x_row)
            if new_x_machine is None:
                logger.debug("X change ignored, no value from UI.")
                return

            logger.debug(f"Handling X change to {new_x_machine}")
            current_y_machine = self.y_row.get_value()
            self.editor.transform.set_position(
                self.items, new_x_machine, current_y_machine
            )
        finally:
            self._in_update = False
            logger.debug("_on_x_changed finished.")

    def _on_y_changed(self, spin_row, GParamSpec):
        logger.debug(f"_on_y_changed called. _in_update={self._in_update}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_y_machine = get_spinrow_float(self.y_row)
            if new_y_machine is None:
                logger.debug("Y change ignored, no value from UI.")
                return

            logger.debug(f"Handling Y change to {new_y_machine}")
            current_x_machine = self.x_row.get_value()
            self.editor.transform.set_position(
                self.items, current_x_machine, new_y_machine
            )
        finally:
            self._in_update = False
            logger.debug("_on_y_changed finished.")

    def _on_angle_changed(self, spin_row, GParamSpec):
        logger.debug(f"_on_angle_changed called. _in_update={self._in_update}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_angle_from_ui = spin_row.get_value()
            new_angle = -new_angle_from_ui
            logger.debug(f"Handling angle change to {new_angle}")

            self.editor.transform.set_angle(self.items, new_angle)
        finally:
            self._in_update = False
            logger.debug("_on_angle_changed finished.")

    def _on_shear_changed(self, spin_row, GParamSpec):
        logger.debug(f"_on_shear_changed called. _in_update={self._in_update}")
        if self._in_update or not self.items:
            return
        self._in_update = True
        try:
            new_shear_from_ui = spin_row.get_value()
            logger.debug(f"Handling shear change to {new_shear_from_ui}")
            self.editor.transform.set_shear(self.items, new_shear_from_ui)
        finally:
            self._in_update = False
            logger.debug("_on_shear_changed finished.")

    def _on_fixed_ratio_toggled(self, switch_row, GParamSpec):
        is_ratio_lockable = self.items and isinstance(
            self.items[0], (WorkPiece, StockItem, Group)
        )
        if not is_ratio_lockable:
            switch_row.set_sensitive(False)
        else:
            switch_row.set_sensitive(True)

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
                        fixed_ratio=False,
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

            new_width, new_height = current_width, current_height

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
            else:
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
            first_item_size = self.items[0].size
            target_y_machine = machine_height - first_item_size[1]

        for item in self.items:
            pos_world, size_world = item.pos, item.size
            target_y_world = 0.0
            if y_axis_down:
                machine_height = bounds[1]
                target_y_world = machine_height - size_world[1]

            if abs(pos_world[1] - target_y_world) >= 1e-9:
                items_to_reset.append(item)

        if items_to_reset:
            current_x_machine = self.x_row.get_value()
            self.editor.transform.set_position(
                items_to_reset, current_x_machine, target_y_machine
            )
