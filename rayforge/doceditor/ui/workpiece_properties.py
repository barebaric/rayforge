import logging
from gi.repository import Gtk, Adw  # type: ignore
from typing import Optional, Tuple, List
from ...shared.ui.expander import Expander  # Import the new custom expander
from ...config import config
from ...core.workpiece import WorkPiece
from ...shared.util.adwfix import get_spinrow_float
from ...undo import (
    ChangePropertyCommand,
    SetterCommand,
)


logger = logging.getLogger(__name__)
default_dim = 100, 100


class WorkpiecePropertiesWidget(Expander):
    def __init__(
        self,
        workpieces: Optional[List[WorkPiece]] = None,
        *args,
        **kwargs,
    ):
        # Initialize the parent Expander widget
        super().__init__(*args, **kwargs)

        self.workpieces = workpieces or []
        self._in_update = False

        # Set the title and default state on the Expander itself
        self.set_title(_("Workpiece Properties"))
        self.set_expanded(True)  # Expanded by default

        # Create a ListBox to hold all the property rows. This replaces the
        # Adw.ExpanderRow's internal list.
        rows_container = Gtk.ListBox()
        rows_container.set_selection_mode(Gtk.SelectionMode.NONE)
        self.set_child(rows_container)

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

        # Fixed Ratio Switch
        self.fixed_ratio_switch = Adw.SwitchRow(
            title=_("Fixed Ratio"), active=True
        )
        self.fixed_ratio_switch.connect(
            "notify::active", self._on_fixed_ratio_toggled
        )
        rows_container.append(self.fixed_ratio_switch)

        # Natural Size Label
        self.natural_size_row = Adw.ActionRow(title=_("Natural Size"))
        self.natural_size_label = Gtk.Label(label=_("N/A"), xalign=0)
        self.natural_size_row.add_suffix(self.natural_size_label)
        rows_container.append(self.natural_size_row)

        # Angle Entry
        self.angle_row = Adw.SpinRow(
            title=_("Angle"),
            adjustment=Gtk.Adjustment.new(0, -360, 360, 1, 10, 0),
            digits=2,
        )
        self.angle_row.connect("notify::value", self._on_angle_changed)
        rows_container.append(self.angle_row)

        # Reset Buttons Row
        self.reset_buttons_row = Adw.ActionRow(title=("Reset"))
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        button_box.set_valign(Gtk.Align.CENTER)

        self.reset_size_button = Gtk.Button(label=_(" Size"))
        self.reset_size_button.connect("clicked", self._on_reset_size_clicked)
        button_box.append(self.reset_size_button)

        self.reset_angle_button = Gtk.Button(label=_("Angle"))
        self.reset_angle_button.connect(
            "clicked", self._on_reset_angle_clicked
        )
        button_box.append(self.reset_angle_button)

        self.reset_aspect_button = Gtk.Button(label=_("Aspect"))
        self.reset_aspect_button.connect(
            "clicked", self._on_reset_aspect_clicked
        )
        button_box.append(self.reset_aspect_button)

        self.reset_buttons_row.add_suffix(button_box)
        rows_container.append(self.reset_buttons_row)

        for workpiece in self.workpieces:
            workpiece.size_changed.connect(self._on_workpiece_size_changed)
            workpiece.pos_changed.connect(self._on_workpiece_pos_changed)
            workpiece.angle_changed.connect(self._on_workpiece_angle_changed)
        self._update_ui_from_workpieces()

    def _calculate_new_size_with_ratio(
        self, workpiece: WorkPiece, value: float, changed_dim: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculates new width and height maintaining aspect ratio."""
        if not workpiece:
            return None, None
        aspect_ratio = workpiece.get_current_aspect_ratio()
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

    def _calculate_resize_anchor_pos(
        self, workpiece: WorkPiece, new_size: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Calculates new canonical model position to keep resize anchored.
        This method operates on canonical Y-up coordinates.
        """
        new_width, new_height = new_size
        bounds = config.machine.dimensions if config.machine else default_dim
        old_pos = workpiece.pos or (0, 0)
        old_w, old_h = (
            workpiece.get_current_size() or workpiece.get_default_size(*bounds)
        )
        old_x, old_y = old_pos

        if workpiece.angle == 0:
            # Anchor is top-left of the object.
            # In a Y-up system, pos is bottom-left, so we must adjust Y.
            new_x = old_x
            new_y = old_y + old_h - new_height
        else:
            # Resize from center for rotated objects.
            new_x = old_x + (old_w - new_width) / 2
            new_y = old_y + (old_h - new_height) / 2
        return new_x, new_y

    def _apply_resize_single(
        self, workpiece: WorkPiece, new_size: Tuple[float, float]
    ):
        """
        Creates a transaction to resize a single workpiece, adjusting its
        position to keep it anchored appropriately.
        """
        if not workpiece or not workpiece.doc:
            return

        new_width, new_height = new_size
        old_pos = workpiece.pos or (0, 0)
        old_size = workpiece.size or (0, 0)

        # Calculate new canonical position based on resize anchor
        new_pos = self._calculate_resize_anchor_pos(workpiece, new_size)

        history = workpiece.doc.history_manager
        with history.transaction(_("Resize workpiece")) as t:
            if new_pos != old_pos:
                pos_cmd = SetterCommand(
                    workpiece, "set_pos", new_args=new_pos, old_args=old_pos
                )
                t.execute(pos_cmd)

            if new_size != old_size:
                size_cmd = SetterCommand(
                    workpiece,
                    "set_size",
                    new_args=new_size,
                    old_args=old_size,
                )
                t.execute(size_cmd)

    def _on_width_changed(self, spin_row, GParamSpec):
        logger.debug(f"Width changed to {spin_row.get_value()}")
        if self._in_update or not self.workpieces:
            return
        self._in_update = True
        try:
            new_width_from_ui = get_spinrow_float(self.width_row)
            if new_width_from_ui is None:
                return

            # Use the first workpiece to update the UI height if ratio is fixed
            if self.fixed_ratio_switch.get_active():
                first_workpiece = self.workpieces[0]
                w, h = self._calculate_new_size_with_ratio(
                    first_workpiece, new_width_from_ui, "width"
                )
                if w is not None and h is not None:
                    self.height_row.set_value(h)
                    self.width_row.set_value(w)

            # Now apply to all workpieces
            for workpiece in self.workpieces:
                new_width = new_width_from_ui
                current_size = workpiece.size
                new_height = (
                    current_size[1]
                    if current_size
                    else self.height_row.get_adjustment().get_lower()
                )

                if self.fixed_ratio_switch.get_active():
                    w, h = self._calculate_new_size_with_ratio(
                        workpiece, new_width, "width"
                    )
                    if w is not None and h is not None:
                        new_width, new_height = w, h

                self._apply_resize_single(workpiece, (new_width, new_height))
        finally:
            self._in_update = False

    def _on_height_changed(self, spin_row, GParamSpec):
        logger.debug(f"Height changed to {spin_row.get_value()}")
        if self._in_update or not self.workpieces:
            return
        self._in_update = True
        try:
            new_height_from_ui = get_spinrow_float(self.height_row)
            if new_height_from_ui is None:
                return

            # Use the first workpiece to update the UI width if ratio is fixed
            if self.fixed_ratio_switch.get_active():
                first_workpiece = self.workpieces[0]
                w, h = self._calculate_new_size_with_ratio(
                    first_workpiece, new_height_from_ui, "height"
                )
                if w is not None and h is not None:
                    self.width_row.set_value(w)
                    self.height_row.set_value(h)

            # Now apply to all workpieces
            for workpiece in self.workpieces:
                new_height = new_height_from_ui
                current_size = workpiece.size
                new_width = (
                    current_size[0]
                    if current_size
                    else self.width_row.get_adjustment().get_lower()
                )

                if self.fixed_ratio_switch.get_active():
                    w, h = self._calculate_new_size_with_ratio(
                        workpiece, new_height, "height"
                    )
                    if w is not None and h is not None:
                        new_width, new_height = w, h

                self._apply_resize_single(workpiece, (new_width, new_height))
        finally:
            self._in_update = False

    def _on_x_changed(self, spin_row, GParamSpec):
        logger.debug(f"X position changed to {spin_row.get_value()}")
        if self._in_update or not self.workpieces:
            return
        self._in_update = True
        try:
            new_x = get_spinrow_float(self.x_row)
            if new_x is None:
                return
            doc = self.workpieces[0].doc
            if not doc:
                return

            with doc.history_manager.transaction(_("Move workpiece")) as t:
                for workpiece in self.workpieces:
                    pos_machine = workpiece.pos_machine or (0.0, 0.0)
                    new_pos = (new_x, pos_machine[1])
                    cmd = ChangePropertyCommand(
                        workpiece, "pos_machine", new_pos
                    )
                    t.execute(cmd)
        finally:
            self._in_update = False

    def _on_y_changed(self, spin_row, GParamSpec):
        logger.debug(f"Y position changed to {spin_row.get_value()}")
        if self._in_update or not self.workpieces:
            return
        self._in_update = True
        try:
            new_y = get_spinrow_float(self.y_row)
            if new_y is None:
                return
            doc = self.workpieces[0].doc
            if not doc:
                return

            with doc.history_manager.transaction(_("Move workpiece")) as t:
                for workpiece in self.workpieces:
                    pos_machine = workpiece.pos_machine or (0.0, 0.0)
                    new_pos = (pos_machine[0], new_y)
                    cmd = ChangePropertyCommand(
                        workpiece, "pos_machine", new_pos
                    )
                    t.execute(cmd)
        finally:
            self._in_update = False

    def _on_angle_changed(self, spin_row, GParamSpec):
        if self._in_update or not self.workpieces:
            return
        self._in_update = True
        try:
            new_angle = spin_row.get_value()
            doc = self.workpieces[0].doc
            if not doc:
                for workpiece in self.workpieces:
                    workpiece.set_angle(new_angle)
                return

            # Group all angle changes into a single transaction for one undo
            with doc.history_manager.transaction(
                _("Change workpiece angle")
            ) as t:
                for workpiece in self.workpieces:
                    cmd = ChangePropertyCommand(
                        workpiece,
                        "angle",
                        new_angle,
                        setter_method_name="set_angle",
                    )
                    t.execute(cmd)
        finally:
            self._in_update = False

    def _on_fixed_ratio_toggled(self, switch_row, GParamSpec):
        logger.debug(f"Fixed ratio toggled: {switch_row.get_active()}")
        return False

    def _on_reset_aspect_clicked(self, button):
        if not self.workpieces:
            return
        doc = self.workpieces[0].doc
        if not doc:
            return

        with doc.history_manager.transaction(
            _("Reset workpiece aspect ratio")
        ) as t:
            for workpiece in self.workpieces:
                current_size = workpiece.get_current_size()
                if not current_size:
                    continue

                current_width = current_size[0]
                default_aspect = workpiece.get_default_aspect_ratio()
                if not default_aspect or default_aspect == 0:
                    continue

                new_height = current_width / default_aspect
                new_size = (current_width, new_height)

                if new_size == current_size:
                    continue

                old_pos = workpiece.pos or (0, 0)
                old_size = workpiece.size or (0, 0)

                # Recalculate position to keep anchor
                new_pos = self._calculate_resize_anchor_pos(
                    workpiece, new_size
                )

                if new_pos != old_pos:
                    pos_cmd = SetterCommand(
                        workpiece,
                        "set_pos",
                        new_args=new_pos,
                        old_args=old_pos,
                    )
                    t.execute(pos_cmd)

                if new_size != old_size:
                    size_cmd = SetterCommand(
                        workpiece,
                        "set_size",
                        new_args=new_size,
                        old_args=old_size,
                    )
                    t.execute(size_cmd)

    def _on_reset_size_clicked(self, button):
        if not self.workpieces:
            return False
        doc = self.workpieces[0].doc
        if not doc:
            return False

        with doc.history_manager.transaction(_("Reset workpiece size")) as t:
            bounds = (
                config.machine.dimensions if config.machine else default_dim
            )
            for workpiece in self.workpieces:
                old_size = workpiece.size or (0, 0)
                old_pos = workpiece.pos or (0, 0)
                natural_width, natural_height = workpiece.get_default_size(
                    *bounds
                )
                new_size = (natural_width, natural_height)

                if new_size == old_size:
                    continue

                new_pos = self._calculate_resize_anchor_pos(
                    workpiece, new_size
                )

                if new_pos != old_pos:
                    pos_cmd = SetterCommand(
                        workpiece,
                        "set_pos",
                        new_args=new_pos,
                        old_args=old_pos,
                    )
                    t.execute(pos_cmd)

                cmd = SetterCommand(
                    workpiece,
                    "set_size",
                    new_args=new_size,
                    old_args=old_size,
                )
                t.execute(cmd)
        return False

    def _on_reset_angle_clicked(self, button):
        if not self.workpieces:
            return
        doc = self.workpieces[0].doc
        if not doc:
            return

        with doc.history_manager.transaction(_("Reset workpiece angle")) as t:
            for workpiece in self.workpieces:
                if workpiece.angle == 0.0:
                    continue
                cmd = ChangePropertyCommand(
                    workpiece,
                    "angle",
                    0.0,
                    setter_method_name="set_angle",
                )
                t.execute(cmd)

    def _on_workpiece_size_changed(self, workpiece):
        if self._in_update:
            return
        logger.debug(f"Workpiece size changed: {workpiece.size}")
        self._update_ui_from_workpieces()

    def set_workpieces(self, workpieces: Optional[List[WorkPiece]]):
        self._in_update = True
        for workpiece in self.workpieces:
            workpiece.size_changed.disconnect(self._on_workpiece_size_changed)
            workpiece.pos_changed.disconnect(self._on_workpiece_pos_changed)
            workpiece.angle_changed.disconnect(
                self._on_workpiece_angle_changed
            )
        self.workpieces = workpieces or []

        # Update the subtitle with the number of selected items
        count = len(self.workpieces)
        if count == 1:
            self.set_subtitle(_("1 item selected"))
        else:
            # This format is standard for gettext pluralization support.
            self.set_subtitle(_(f"{count} items selected"))

        for workpiece in self.workpieces:
            workpiece.size_changed.connect(self._on_workpiece_size_changed)
            workpiece.pos_changed.connect(self._on_workpiece_pos_changed)
            workpiece.angle_changed.connect(self._on_workpiece_angle_changed)
        self._in_update = False
        self._update_ui_from_workpieces()

    def _on_workpiece_pos_changed(self, workpiece):
        if self._in_update:
            return
        logger.debug(f"Workpiece position changed: {workpiece.pos}")
        self._update_ui_from_workpieces()

    def _on_workpiece_angle_changed(self, workpiece):
        if self._in_update:
            return
        logger.debug(f"Workpiece angle changed: {workpiece.angle}")
        self._update_ui_from_workpieces()

    def _update_ui_from_workpieces(self):
        logger.debug(f"Updating UI for workpieces: {self.workpieces}")
        if not self.workpieces:
            self.set_sensitive(False)
            return

        self.set_sensitive(True)
        workpiece = self.workpieces[0]

        self._in_update = True
        bounds = config.machine.dimensions if config.machine else default_dim
        y_axis_down = config.machine.y_axis_down if config.machine else False
        size = workpiece.get_current_size() or workpiece.get_default_size(
            *bounds
        )
        pos = workpiece.pos_machine  # Use machine-native coordinates
        angle = workpiece.angle

        # Update Y-axis label to reflect coordinate system
        if y_axis_down:
            self.y_row.set_subtitle(_("Zero is at the top"))
        else:
            self.y_row.set_subtitle(_("Zero is at the bottom"))

        if size:
            width, height = size
            logger.debug(f"Updating UI: width={width}, height={height}")
            self.width_row.set_value(width)
            self.height_row.set_value(height)
            natural_width, natural_height = workpiece.get_default_size(*bounds)
            self.natural_size_label.set_label(
                f"{natural_width:.2f}x{natural_height:.2f}"
            )
        else:
            self.natural_size_label.set_label("N/A")
        if pos:
            x, y = pos
            logger.debug(f"Updating UI: x={x}, y={y}")
            self.x_row.set_value(x)
            self.y_row.set_value(y)

        logger.debug(f"Updating UI: angle={angle}")
        self.angle_row.set_value(angle)

        self._in_update = False
