import logging
from gi.repository import Gtk, Adw, Gdk  # type: ignore
from typing import Optional, Tuple
from ..config import config
from ..models.workpiece import WorkPiece
from ..util.adwfix import get_spinrow_float
from ..undo import (
    ChangePropertyCommand,
    SetterCommand,
)


css = """
.workpiece-properties .boxed-list {
    margin: 0 0 12px 0;
    box-shadow: 0 8px 8px rgba(0, 0, 0, 0.1);
}
"""


logger = logging.getLogger(__name__)


class WorkpiecePropertiesWidget(Adw.PreferencesGroup):
    def __init__(
        self,
        workpiece: Optional[WorkPiece] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.add_css_class("workpiece-properties")
        self.apply_css()
        self.workpiece = workpiece
        self._in_update = False
        self.set_title(_("Workpiece Properties"))

        # X Position Entry
        self.x_row = Adw.SpinRow(
            title=_("X Position"),
            adjustment=Gtk.Adjustment.new(0, -10000, 10000, 1.0, 1, 0),
        )
        self.x_row.set_digits(2)
        self.x_row.connect("notify::value", self._on_x_changed)
        self.add(self.x_row)

        # Y Position Entry
        self.y_row = Adw.SpinRow(
            title=_("Y Position"),
            adjustment=Gtk.Adjustment.new(0, -10000, 10000, 1.0, 1, 0),
        )
        self.y_row.set_digits(2)
        self.y_row.connect("notify::value", self._on_y_changed)
        self.add(self.y_row)

        # Width Entry
        self.width_row = Adw.SpinRow(
            title=_("Width"),
            adjustment=Gtk.Adjustment.new(10, 1, 10000, 1.0, 1, 0),
        )
        self.width_row.set_digits(2)
        self.width_row.connect("notify::value", self._on_width_changed)
        self.add(self.width_row)

        # Height Entry
        self.height_row = Adw.SpinRow(
            title=_("Height"),
            adjustment=Gtk.Adjustment.new(10, 1, 10000, 1.0, 1, 0),
        )
        self.height_row.set_digits(2)
        self.height_row.connect("notify::value", self._on_height_changed)
        self.add(self.height_row)

        # Fixed Ratio Switch
        self.fixed_ratio_switch = Adw.SwitchRow(
            title=_("Fixed Ratio"), active=True
        )
        self.fixed_ratio_switch.connect(
            "notify::active", self._on_fixed_ratio_toggled
        )
        self.add(self.fixed_ratio_switch)

        # Natural Size Label
        self.natural_size_row = Adw.ActionRow(title=_("Natural Size"))
        self.natural_size_label = Gtk.Label(label=_("N/A"), xalign=0)
        self.natural_size_row.add_suffix(self.natural_size_label)
        self.add(self.natural_size_row)

        # Reset Size Button
        self.reset_row = Adw.ActionRow(title=_("Reset Size"))
        self.reset_button = Gtk.Button(label=_("Reset"))
        self.reset_button.set_halign(Gtk.Align.END)
        self.reset_button.set_valign(Gtk.Align.CENTER)
        self.reset_button.connect("clicked", self._on_reset_clicked)
        self.reset_row.add_suffix(self.reset_button)
        self.reset_row.activatable_widget = self.reset_button
        self.add(self.reset_row)

        # Angle Entry
        self.angle_row = Adw.SpinRow(
            title=_("Angle"),
            adjustment=Gtk.Adjustment.new(0, -360, 360, 1, 10, 0),
            digits=2,
        )
        self.angle_row.connect("notify::value", self._on_angle_changed)
        self.add(self.angle_row)

        # Reset Angle Button
        self.reset_angle_row = Adw.ActionRow(title=_("Reset Angle"))
        self.reset_angle_button = Gtk.Button(label=_("Reset"))
        self.reset_angle_button.set_halign(Gtk.Align.END)
        self.reset_angle_button.set_valign(Gtk.Align.CENTER)
        self.reset_angle_button.connect(
            "clicked", self._on_reset_angle_clicked
        )
        self.reset_angle_row.add_suffix(self.reset_angle_button)
        self.reset_angle_row.activatable_widget = self.reset_angle_button
        self.add(self.reset_angle_row)

        if self.workpiece:
            self.workpiece.size_changed.connect(
                self._on_workpiece_size_changed
            )
            self.workpiece.pos_changed.connect(self._on_workpiece_pos_changed)
            self.workpiece.angle_changed.connect(
                self._on_workpiece_angle_changed
            )
        self._update_ui_from_workpiece()

    def apply_css(self):
        provider = Gtk.CssProvider()
        provider.load_from_string(css)
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _commit_pos_transaction(self, new_pos: Tuple[float, float]):
        """Commits workpiece position changes to the history manager."""
        if not self.workpiece:
            return

        old_pos = self.workpiece.pos or (0, 0)
        doc = self.workpiece.doc
        if not doc:
            self.workpiece.set_pos(*new_pos)
            return

        cmd = SetterCommand(
            self.workpiece,
            "set_pos",
            new_args=new_pos,
            old_args=old_pos,
            name=_("Move the workpiece"),
        )
        doc.history_manager.execute(cmd)

    def _calculate_new_size_with_ratio(
        self, value: float, changed_dim: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculates new width and height maintaining aspect ratio."""
        if not self.workpiece:
            return None, None
        aspect_ratio = self.workpiece.get_current_aspect_ratio()
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

    def _commit_resize_transaction(self, new_size: Tuple[float, float]):
        """Calculates new position and commits size/pos changes."""
        if not self.workpiece:
            return

        new_width, new_height = new_size
        bounds = config.machine.dimensions
        old_pos = self.workpiece.pos or (0, 0)
        old_size = self.workpiece.size or (0, 0)
        old_w, old_h = (
            self.workpiece.get_current_size()
            or self.workpiece.get_default_size(*bounds)
        )
        old_x, old_y = old_pos

        if self.workpiece.angle == 0:
            # Resize from top-left for un-rotated
            new_x = old_x
            new_y = old_y + old_h - new_height
        else:
            # Resize from center for rotated
            new_x = old_x + (old_w - new_width) / 2
            new_y = old_y + (old_h - new_height) / 2

        doc = self.workpiece.doc
        if not doc:
            self.workpiece.set_pos(new_x, new_y)
            self.workpiece.set_size(new_width, new_height)
            return

        history = doc.history_manager
        history.begin_transaction(_("Resize"))
        try:
            pos_cmd = SetterCommand(
                self.workpiece,
                "set_pos",
                new_args=(new_x, new_y),
                old_args=old_pos,
                name=_("Move the workpiece"),
            )
            history.execute(pos_cmd)
            size_cmd = SetterCommand(
                self.workpiece,
                "set_size",
                new_args=(new_width, new_height),
                old_args=old_size,
                name=_("Resize the workpiece"),
            )
            history.execute(size_cmd)
        finally:
            history.end_transaction()

    def _on_width_changed(self, spin_row, GParamSpec):
        logger.debug(f"Width changed to {spin_row.get_value()}")
        if self._in_update or not self.workpiece:
            return
        self._in_update = True
        try:
            new_width = get_spinrow_float(self.width_row)
            if new_width is None:
                return

            current_size = self.workpiece.size
            new_height = (
                current_size[1]
                if current_size
                else self.height_row.get_adjustment().get_lower()
            )

            if self.fixed_ratio_switch.get_active():
                w, h = self._calculate_new_size_with_ratio(new_width, "width")
                if w is not None and h is not None:
                    new_width, new_height = w, h
                    self.height_row.set_value(new_height)
                    self.width_row.set_value(new_width)

            self._commit_resize_transaction((new_width, new_height))
        finally:
            self._in_update = False

    def _on_height_changed(self, spin_row, GParamSpec):
        logger.debug(f"Height changed to {spin_row.get_value()}")
        if self._in_update or not self.workpiece:
            return
        self._in_update = True
        try:
            new_height = get_spinrow_float(self.height_row)
            if new_height is None:
                return

            current_size = self.workpiece.size
            new_width = (
                current_size[0]
                if current_size
                else self.width_row.get_adjustment().get_lower()
            )

            if self.fixed_ratio_switch.get_active():
                w, h = self._calculate_new_size_with_ratio(
                    new_height, "height"
                )
                if w is not None and h is not None:
                    new_width, new_height = w, h
                    self.width_row.set_value(new_width)
                    self.height_row.set_value(new_height)

            self._commit_resize_transaction((new_width, new_height))
        finally:
            self._in_update = False

    def _on_x_changed(self, spin_row, GParamSpec):
        logger.debug(f"X position changed to {spin_row.get_value()}")
        if self._in_update or not self.workpiece:
            return
        self._in_update = True
        try:
            new_x = get_spinrow_float(self.x_row)
            if new_x is None:
                return

            old_pos = self.workpiece.pos
            new_y = old_pos[1] if old_pos else 0.0
            self._commit_pos_transaction((new_x, new_y))
        finally:
            self._in_update = False

    def _on_y_changed(self, spin_row, GParamSpec):
        logger.debug(f"Y position changed to {spin_row.get_value()}")
        if self._in_update or not self.workpiece:
            return
        self._in_update = True
        try:
            new_y = get_spinrow_float(self.y_row)
            if new_y is None:
                return

            old_pos = self.workpiece.pos
            new_x = old_pos[0] if old_pos else 0.0
            self._commit_pos_transaction((new_x, new_y))
        finally:
            self._in_update = False

    def _on_angle_changed(self, spin_row, GParamSpec):
        if self._in_update or not self.workpiece:
            return
        self._in_update = True
        try:
            doc = self.workpiece.doc
            if not doc:
                self.workpiece.set_angle(spin_row.get_value())
                return

            cmd = ChangePropertyCommand(
                self.workpiece,
                "angle",
                spin_row.get_value(),
                setter_method_name="set_angle",
                name=_("Change workpiece angle"),
            )
            doc.history_manager.execute(cmd)
        finally:
            self._in_update = False

    def _on_fixed_ratio_toggled(self, switch_row, GParamSpec):
        """
        This function's only purpose is to allow the user to toggle the
        switch state. It does not perform any action itself and does not
        need an undo entry. The width/height change handlers are
        responsible for reading this switch's state.
        """
        logger.debug(f"Fixed ratio toggled: {switch_row.get_active()}")
        # No action is needed here. The event is captured simply to
        # allow the state of the switch to be changed by user interaction.
        return False

    def _on_reset_clicked(self, button):
        if not self.workpiece:
            return False

        bounds = config.machine.dimensions
        old_size = self.workpiece.size
        natural_width, natural_height = self.workpiece.get_default_size(
            *bounds
        )

        if not old_size or not self.workpiece.doc:
            self.workpiece.set_size(natural_width, natural_height)
            self._update_ui_from_workpiece()
            return False

        cmd = SetterCommand(
            self.workpiece,
            "set_size",
            new_args=(natural_width, natural_height),
            old_args=old_size,
            name=_("Reset workpiece size"),
        )
        self.workpiece.doc.history_manager.execute(cmd)
        self._update_ui_from_workpiece()

    def _on_reset_angle_clicked(self, button):
        if not self.workpiece:
            return
        if not self.workpiece.doc:
            self.workpiece.set_angle(0.0)
            return

        cmd = ChangePropertyCommand(
            self.workpiece,
            "angle",
            0.0,
            setter_method_name="set_angle",
            name=_("Reset workpiece angle"),
        )
        self.workpiece.doc.history_manager.execute(cmd)

    def _on_workpiece_size_changed(self, workpiece):
        if self._in_update:
            return
        logger.debug(f"Workpiece size changed: {workpiece.size}")
        self._update_ui_from_workpiece()

    def set_workpiece(self, workpiece: Optional[WorkPiece]):
        self._in_update = True
        if self.workpiece:
            self.workpiece.size_changed.disconnect(
                self._on_workpiece_size_changed
            )
            self.workpiece.pos_changed.disconnect(
                self._on_workpiece_pos_changed
            )
            self.workpiece.angle_changed.disconnect(
                self._on_workpiece_angle_changed
            )
        self.workpiece = workpiece
        if self.workpiece:
            self.workpiece.size_changed.connect(
                self._on_workpiece_size_changed
            )
            self.workpiece.pos_changed.connect(self._on_workpiece_pos_changed)
            self.workpiece.angle_changed.connect(
                self._on_workpiece_angle_changed
            )
        self._in_update = False
        self._update_ui_from_workpiece()

    def _on_workpiece_pos_changed(self, workpiece):
        if self._in_update:
            return
        logger.debug(f"Workpiece position changed: {workpiece.pos}")
        self._update_ui_from_workpiece()

    def _on_workpiece_angle_changed(self, workpiece):
        if self._in_update:
            return
        logger.debug(f"Workpiece angle changed: {workpiece.angle}")
        self._update_ui_from_workpiece()

    def _update_ui_from_workpiece(self):
        logger.debug(f"Updating UI for workpiece: {self.workpiece}")
        if not self.workpiece:
            return
        self._in_update = True
        bounds = config.machine.dimensions
        size = (
            self.workpiece.get_current_size()
            or self.workpiece.get_default_size(*bounds)
        )
        pos = self.workpiece.pos
        angle = self.workpiece.angle

        if size:
            width, height = size
            logger.debug(f"Updating UI: width={width}, height={height}")
            self.width_row.set_value(width)
            self.height_row.set_value(height)
            natural_width, natural_height = self.workpiece.get_default_size(
                *bounds
            )
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
