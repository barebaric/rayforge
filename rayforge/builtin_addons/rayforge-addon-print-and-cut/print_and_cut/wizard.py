import math
import logging
from gettext import gettext as _
from typing import Optional, Tuple

from gi.repository import Adw, Gtk

from rayforge.core.matrix import Matrix
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Machine
from rayforge.machine.cmd import MachineCmd
from rayforge.doceditor.editor import DocEditor
from rayforge.ui_gtk.shared.patched_dialog_window import (
    PatchedDialogWindow,
)
from rayforge.ui_gtk.icons import get_icon
from rayforge.ui_gtk.machine.jog_widget import JogWidget
from rayforge.ui_gtk.shared.adwfix import get_spinrow_float
from .pick_surface import PickSurface

logger = logging.getLogger(__name__)

MIN_POINT_DISTANCE = 0.5

_session_state: dict = {}


def calculate_alignment_transform(
    d1: Tuple[float, float],
    d2: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    allow_scale: bool = False,
) -> Matrix:
    angle_d = math.atan2(d2[1] - d1[1], d2[0] - d1[0])
    angle_p = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle = math.degrees(angle_p - angle_d)

    dist_d = math.hypot(d2[0] - d1[0], d2[1] - d1[1])
    dist_p = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    scale = dist_p / dist_d if allow_scale and dist_d > 0 else 1.0

    return (
        Matrix.translation(*p1)
        @ Matrix.rotation(angle)
        @ Matrix.scale(scale, scale)
        @ Matrix.translation(-d1[0], -d1[1])
    )


class PrintAndCutWizard(PatchedDialogWindow):
    def __init__(
        self,
        parent,
        workpiece: WorkPiece,
        machine: Machine,
        machine_cmd: MachineCmd,
        editor: DocEditor,
        **kwargs,
    ):
        super().__init__(
            transient_for=parent,
            default_width=1150,
            default_height=750,
            title=_("Print & Cut"),
            **kwargs,
        )

        self._workpiece = workpiece
        self._machine = machine
        self._machine_cmd = machine_cmd
        self._editor = editor

        self._design_point1: Optional[Tuple[float, float]] = None
        self._design_point2: Optional[Tuple[float, float]] = None

        self._physical_point1: Optional[Tuple[float, float]] = None
        self._physical_point2: Optional[Tuple[float, float]] = None

        self._allow_scale: bool = False

        self._setup_ui()
        self._restore_session_state()

    def _setup_ui(self):
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        header = Adw.HeaderBar()
        content.append(header)

        self._main_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=16,
            margin_start=12,
            margin_top=12,
            margin_bottom=12,
        )
        content.append(self._main_box)

        self._left_panel = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
        )
        self._left_panel.set_hexpand(True)
        self._left_panel.set_vexpand(True)
        self._main_box.append(self._left_panel)

        self._right_stack = Gtk.Stack()
        self._right_stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self._right_stack.set_hexpand(False)
        self._main_box.append(self._right_stack)

        self._setup_left_panel()
        self._setup_right_stack()

        self._button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_end=12,
            margin_bottom=12,
        )
        content.append(self._button_box)

        self._back_btn = Gtk.Button(label=_("Back"))
        self._back_btn.add_css_class("flat")
        self._back_btn.connect("clicked", self._on_back_clicked)
        self._back_btn.set_visible(False)
        self._button_box.append(self._back_btn)

        self._reset_btn = Gtk.Button(label=_("Reset"))
        self._reset_btn.add_css_class("flat")
        self._reset_btn.connect("clicked", self._on_reset_clicked)
        self._button_box.append(self._reset_btn)

        self._cancel_btn = Gtk.Button(label=_("Cancel"))
        self._cancel_btn.add_css_class("flat")
        self._cancel_btn.connect("clicked", lambda _: self.close())
        self._button_box.append(self._cancel_btn)

        self._next_btn = Gtk.Button(label=_("Next"))
        self._next_btn.add_css_class("suggested-action")
        self._next_btn.connect("clicked", self._on_next_clicked)
        self._next_btn.set_sensitive(False)
        self._button_box.append(self._next_btn)

        self._apply_btn = Gtk.Button(label=_("Apply"))
        self._apply_btn.add_css_class("suggested-action")
        self._apply_btn.connect("clicked", self._on_apply_clicked)
        self._apply_btn.set_visible(False)
        self._button_box.append(self._apply_btn)

        self._right_stack.connect(
            "notify::visible-child", self._on_page_changed
        )

    def _setup_left_panel(self):
        self._pick_surface = PickSurface(
            workpiece=self._workpiece,
        )
        self._pick_surface.set_hexpand(True)
        self._pick_surface.set_vexpand(True)
        self._pick_surface.set_halign(Gtk.Align.FILL)
        self._pick_surface.point_picked.connect(self._on_design_point_picked)
        self._pick_surface.points_reset.connect(self._on_design_points_reset)
        self._pick_surface.points_changed.connect(
            self._on_design_points_changed
        )
        self._left_panel.append(self._pick_surface)

    def _setup_right_stack(self):
        self._setup_pick_panel()
        self._setup_jog_panel()
        self._setup_apply_panel()

    def _setup_pick_panel(self):
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._right_stack.add_named(scroll, "pick")

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        box.set_margin_start(12)
        box.set_margin_end(32)
        box.set_margin_top(4)
        box.set_margin_bottom(12)
        scroll.set_child(box)

        intro_group = Adw.PreferencesGroup(
            title=_("Instructions"),
            description=_(
                "Click on two alignment points on the design. "
                "These will be matched to physical locations in "
                "the next step."
            ),
        )
        box.append(intro_group)

        points_group = Adw.PreferencesGroup(title=_("Picked Points"))
        box.append(points_group)

        self._point1_row = Adw.ActionRow(title=_("Point 1"))
        self._point1_row.set_subtitle(_("No point picked"))
        points_group.add(self._point1_row)

        self._point2_row = Adw.ActionRow(title=_("Point 2"))
        self._point2_row.set_subtitle(_("No point picked"))
        points_group.add(self._point2_row)

        self._pick_status_row = Adw.ActionRow(title=_("Status"))
        self._pick_status_row.set_subtitle(
            _("Click on the first alignment point on the design.")
        )
        points_group.add(self._pick_status_row)

    def _setup_jog_panel(self):
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._right_stack.add_named(scroll, "jog")

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        box.set_margin_start(12)
        box.set_margin_end(32)
        box.set_margin_top(4)
        box.set_margin_bottom(12)
        scroll.set_child(box)

        intro_group = Adw.PreferencesGroup(
            title=_("Instructions"),
            description=_(
                "Jog the laser to the physical location of each "
                "point, then record the position."
            ),
        )
        box.append(intro_group)

        jog_frame = Gtk.Frame(halign=Gtk.Align.CENTER)
        jog_frame.add_css_class("card")
        box.append(jog_frame)

        jog_inner = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=6,
            margin_bottom=6,
            margin_start=6,
            margin_end=6,
            halign=Gtk.Align.CENTER,
        )
        jog_frame.set_child(jog_inner)

        self._jog_widget = JogWidget(show_actions=False)
        self._jog_widget.set_machine(self._machine, self._machine_cmd)
        jog_inner.append(self._jog_widget)

        controls_group = Adw.PreferencesGroup()
        box.append(controls_group)

        distance_adjustment = Gtk.Adjustment(
            value=10.0, lower=0.1, upper=1000, step_increment=1
        )
        self._distance_row = Adw.SpinRow(
            title=_("Jog Distance"),
            subtitle=_("Distance in machine units"),
            adjustment=distance_adjustment,
            digits=1,
        )
        self._distance_row.connect("changed", self._on_distance_changed)
        controls_group.add(self._distance_row)

        self._focus_active = False
        self._focus_on_icon = get_icon("laser-on-symbolic")
        self._focus_off_icon = get_icon("laser-off-symbolic")
        self._focus_btn = Gtk.ToggleButton(
            tooltip_text=_("Toggle focus laser"),
            valign=Gtk.Align.CENTER,
        )
        self._focus_btn.set_child(self._focus_on_icon)
        self._focus_btn.connect("toggled", self._on_focus_toggled)

        self._focus_row = Adw.ActionRow(title=_("Focus Laser"))
        self._focus_row.add_suffix(self._focus_btn)
        controls_group.add(self._focus_row)

        head = self._machine.get_default_head() if self._machine else None
        if head:
            head.changed.connect(self._on_head_changed)
        self._update_focus_sensitivity()

        positions_group = Adw.PreferencesGroup(title=_("Positions"))
        box.append(positions_group)

        self._record1_btn = Gtk.Button(
            label=_("Record"), valign=Gtk.Align.CENTER
        )
        self._record1_btn.add_css_class("suggested-action")
        self._record1_btn.connect("clicked", self._on_record_clicked, 0)

        self._goto1_btn = Gtk.Button(
            child=get_icon("zero-here-symbolic"),
            valign=Gtk.Align.CENTER,
            tooltip_text=_("Go to recorded position"),
        )
        self._goto1_btn.add_css_class("flat")
        self._goto1_btn.connect("clicked", self._on_goto_clicked, 0)
        self._goto1_btn.set_sensitive(False)

        self._pos1_row = Adw.ActionRow(title=_("Position 1"))
        self._pos1_row.set_subtitle(_("Not recorded"))
        self._pos1_row.add_suffix(self._goto1_btn)
        self._pos1_row.add_suffix(self._record1_btn)
        positions_group.add(self._pos1_row)

        self._record2_btn = Gtk.Button(
            label=_("Record"), valign=Gtk.Align.CENTER
        )
        self._record2_btn.add_css_class("suggested-action")
        self._record2_btn.connect("clicked", self._on_record_clicked, 1)

        self._goto2_btn = Gtk.Button(
            child=get_icon("zero-here-symbolic"),
            valign=Gtk.Align.CENTER,
            tooltip_text=_("Go to recorded position"),
        )
        self._goto2_btn.add_css_class("flat")
        self._goto2_btn.connect("clicked", self._on_goto_clicked, 1)
        self._goto2_btn.set_sensitive(False)

        self._pos2_row = Adw.ActionRow(title=_("Position 2"))
        self._pos2_row.set_subtitle(_("Not recorded"))
        self._pos2_row.add_suffix(self._goto2_btn)
        self._pos2_row.add_suffix(self._record2_btn)
        positions_group.add(self._pos2_row)

        self._laser_row = Adw.ActionRow(title=_("Laser Position"))
        self._laser_row.set_subtitle(_("X: --- Y: ---"))
        positions_group.add(self._laser_row)

        if self._machine:
            self._machine.state_changed.connect(self._on_machine_state_changed)
            self._machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self._update_connection_sensitive()
            self._update_laser_position()

    def _setup_apply_panel(self):
        scroll = Gtk.ScrolledWindow()
        scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._right_stack.add_named(scroll, "apply")

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            hexpand=True,
        )
        box.set_margin_start(24)
        box.set_margin_end(24)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        scroll.set_child(box)

        intro_group = Adw.PreferencesGroup(
            title=_("Transform Preview"),
            description=_(
                "Review the computed alignment transform before applying it."
            ),
        )
        box.append(intro_group)

        transform_group = Adw.PreferencesGroup(title=_("Transform"))
        box.append(transform_group)

        self._translation_row = Adw.ActionRow(title=_("Translation"))
        self._translation_row.set_subtitle("---")
        transform_group.add(self._translation_row)

        self._rotation_row = Adw.ActionRow(title=_("Rotation"))
        self._rotation_row.set_subtitle("---")
        transform_group.add(self._rotation_row)

        self._scale_row = Adw.ActionRow(title=_("Scale"))
        self._scale_row.set_subtitle("---")
        transform_group.add(self._scale_row)

        self._scale_check = Gtk.CheckButton(
            label=_("Allow scaling"),
        )
        self._scale_check.set_active(False)
        self._scale_check.connect("toggled", self._on_scale_toggled)
        box.append(self._scale_check)

        self._warning_label = Gtk.Label(label="")
        self._warning_label.add_css_class("warning-label")
        self._warning_label.set_visible(False)
        self._warning_label.set_wrap(True)
        self._warning_label.set_xalign(0.0)
        box.append(self._warning_label)

    def _on_design_point_picked(self, sender, **kwargs):
        index = kwargs.get("index", 0)
        x = kwargs.get("x", 0.0)
        y = kwargs.get("y", 0.0)

        if index == 0:
            self._design_point1 = (x, y)
            self._point1_row.set_subtitle(_("Point picked"))
            self._pick_status_row.set_subtitle(
                _("Click on the second alignment point on the design.")
            )
        elif index == 1:
            self._design_point2 = (x, y)
            self._point2_row.set_subtitle(_("Point picked"))
            self._pick_status_row.set_subtitle(
                _("Both points selected. Click Next to continue.")
            )

        self._update_pick_next_btn()
        self._save_session_state()

    def _on_design_points_changed(self, sender):
        p1 = self._pick_surface.point1
        p2 = self._pick_surface.point2
        self._design_point1 = p1
        self._design_point2 = p2
        if p1 is not None:
            self._point1_row.set_subtitle(_("Point picked"))
        if p2 is not None:
            self._point2_row.set_subtitle(_("Point picked"))
        self._update_pick_next_btn()
        self._save_session_state()

    def _on_design_points_reset(self, sender):
        self._design_point1 = None
        self._design_point2 = None
        self._physical_point1 = None
        self._physical_point2 = None
        self._point1_row.set_subtitle(_("No point picked"))
        self._point2_row.set_subtitle(_("No point picked"))
        self._pos1_row.set_subtitle(_("Not recorded"))
        self._pos2_row.set_subtitle(_("Not recorded"))
        self._goto1_btn.set_sensitive(False)
        self._goto2_btn.set_sensitive(False)
        self._pick_status_row.set_subtitle(
            _("Click on the first alignment point on the design.")
        )
        self._update_pick_next_btn()
        self._save_session_state()

    def _update_pick_next_btn(self):
        p1 = self._design_point1
        p2 = self._design_point2
        if p1 is not None and p2 is not None:
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self._next_btn.set_sensitive(dist >= MIN_POINT_DISTANCE)
            if dist < MIN_POINT_DISTANCE:
                self._pick_status_row.set_subtitle(
                    _("Points are too close. Pick points further apart.")
                )
        else:
            self._next_btn.set_sensitive(False)

    def _restore_session_state(self):
        wp_id = self._workpiece.uid
        state = _session_state.get(wp_id)
        if state is None:
            return

        dp1 = state.get("design_point1")
        dp2 = state.get("design_point2")
        pp1 = state.get("physical_point1")
        pp2 = state.get("physical_point2")

        if dp1 is not None or dp2 is not None:
            self._pick_surface.set_points(dp1, dp2)
            self._design_point1 = dp1
            self._design_point2 = dp2
            if dp1 is not None:
                self._point1_row.set_subtitle(_("Point picked"))
            if dp2 is not None:
                self._point2_row.set_subtitle(_("Point picked"))
            if dp1 is not None and dp2 is not None:
                self._pick_status_row.set_subtitle(
                    _("Both points selected. Click Next to continue.")
                )
            elif dp1 is not None:
                self._pick_status_row.set_subtitle(
                    _("Click on the second alignment point on the design.")
                )

        if pp1 is not None:
            self._physical_point1 = pp1
            self._pos1_row.set_subtitle(f"({pp1[0]:.2f}, {pp1[1]:.2f})")
        if pp2 is not None:
            self._physical_point2 = pp2
            self._pos2_row.set_subtitle(f"({pp2[0]:.2f}, {pp2[1]:.2f})")

        connected = self._machine and self._machine.is_connected()
        if pp1 is not None:
            self._goto1_btn.set_sensitive(connected)
        if pp2 is not None:
            self._goto2_btn.set_sensitive(connected)

        self._update_pick_next_btn()

    def _save_session_state(self):
        wp_id = self._workpiece.uid
        _session_state[wp_id] = {
            "design_point1": self._design_point1,
            "design_point2": self._design_point2,
            "physical_point1": self._physical_point1,
            "physical_point2": self._physical_point2,
        }

    def _on_record_clicked(self, button, pos_index):
        if not self._machine or not self._machine.is_connected():
            return

        pos = self._machine.get_current_position()
        if pos is None:
            return

        x_val, y_val, _z_val = pos
        if x_val is None or y_val is None:
            return

        subtitle = f"({x_val:.2f}, {y_val:.2f})"
        if pos_index == 0:
            self._physical_point1 = (x_val, y_val)
            self._pos1_row.set_subtitle(subtitle)
            self._goto1_btn.set_sensitive(True)
        else:
            self._physical_point2 = (x_val, y_val)
            self._pos2_row.set_subtitle(subtitle)
            self._goto2_btn.set_sensitive(True)

        self._update_jog_next_btn()
        self._save_session_state()

    def _on_goto_clicked(self, button, pos_index):
        if not self._machine or not self._machine.is_connected():
            return
        if pos_index == 0:
            point = self._physical_point1
        else:
            point = self._physical_point2
        if point is None:
            return
        self._machine_cmd.move_to(self._machine, point[0], point[1])

    def _update_jog_next_btn(self):
        p1 = self._physical_point1
        p2 = self._physical_point2
        if p1 is not None and p2 is not None:
            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            self._next_btn.set_sensitive(dist >= MIN_POINT_DISTANCE)
        else:
            self._next_btn.set_sensitive(False)

    def _on_machine_state_changed(self, machine, state):
        self._update_laser_position()

    def _on_connection_status_changed(self, sender, **kwargs):
        self._update_connection_sensitive()

    def _update_connection_sensitive(self):
        connected = self._machine and self._machine.is_connected()
        self._record1_btn.set_sensitive(connected)
        self._record2_btn.set_sensitive(connected)
        self._goto1_btn.set_sensitive(
            connected and self._physical_point1 is not None
        )
        self._goto2_btn.set_sensitive(
            connected and self._physical_point2 is not None
        )
        self._update_focus_sensitivity()
        if connected:
            self._update_laser_position()

    def _update_laser_position(self):
        if not self._machine:
            return
        pos = self._machine.get_current_position()
        if pos:
            x_val, y_val, _z_val = pos
            if x_val is not None and y_val is not None:
                self._laser_row.set_subtitle(f"X: {x_val:.2f}  Y: {y_val:.2f}")

    def _on_distance_changed(self, spin_row):
        self._jog_widget.jog_distance = get_spinrow_float(spin_row)

    def _on_focus_toggled(self, button):
        if not self._machine or not self._machine_cmd:
            return
        head = self._machine.get_default_head()
        if not head:
            return
        self._focus_active = button.get_active()
        if self._focus_active:
            self._machine_cmd.set_focus_power(head, head.focus_power_percent)
            button.set_child(self._focus_off_icon)
        else:
            self._machine_cmd.set_focus_power(head, 0)
            button.set_child(self._focus_on_icon)

    def _disable_focus(self):
        if self._focus_active and self._machine and self._machine_cmd:
            head = self._machine.get_default_head()
            if head:
                self._machine_cmd.set_focus_power(head, 0)
                self._focus_active = False
                if self._focus_btn:
                    self._focus_btn.set_active(False)

    def _on_head_changed(self, head, *args):
        self._update_focus_sensitivity()

    def _update_focus_sensitivity(self):
        head = self._machine.get_default_head() if self._machine else None
        if head and head.focus_power_percent > 0:
            self._focus_btn.set_sensitive(True)
            self._focus_row.set_subtitle(
                _("Turn on laser at focus power to locate position")
            )
        else:
            self._focus_row.set_subtitle(
                _("Set a focus power in laser preferences to enable")
            )
            self._focus_btn.set_sensitive(False)
            if self._focus_active:
                self._disable_focus()

    def _on_scale_toggled(self, check_btn):
        self._allow_scale = check_btn.get_active()
        self._update_apply_preview()

    def _local_to_world(
        self, norm_x: float, norm_y: float
    ) -> Tuple[float, float]:
        return self._workpiece.get_world_transform().transform_point(
            (norm_x, norm_y)
        )

    def _machine_to_world(self, wx: float, wy: float) -> Tuple[float, float]:
        space = self._machine.get_coordinate_space()
        wcs_x, wcs_y, _wcs_z = self._machine.get_active_wcs_offset()
        return space.machine_point_to_world(wx + wcs_x, wy + wcs_y)

    def _get_world_design_points(self):
        assert self._design_point1 is not None
        assert self._design_point2 is not None
        d1 = self._local_to_world(*self._design_point1)
        d2 = self._local_to_world(*self._design_point2)
        return d1, d2

    def _get_world_physical_points(self):
        assert self._physical_point1 is not None
        assert self._physical_point2 is not None
        p1 = self._machine_to_world(*self._physical_point1)
        p2 = self._machine_to_world(*self._physical_point2)
        return p1, p2

    def _update_apply_preview(self):
        if (
            self._design_point1 is None
            or self._design_point2 is None
            or self._physical_point1 is None
            or self._physical_point2 is None
        ):
            return

        d1, d2 = self._get_world_design_points()
        p1, p2 = self._get_world_physical_points()

        T = calculate_alignment_transform(
            d1, d2, p1, p2, allow_scale=self._allow_scale
        )

        _tx, _ty, angle, sx, sy, _skew = T.decompose()
        tx, ty = T.get_translation()

        self._translation_row.set_subtitle(f"({tx:.2f}, {ty:.2f})")
        self._rotation_row.set_subtitle(f"{angle:.2f}\u00b0")

        if self._allow_scale:
            self._scale_row.set_subtitle(f"{sx:.4f} x {sy:.4f}")
            self._warning_label.set_visible(False)
        else:
            dist_d = math.hypot(
                d2[0] - d1[0],
                d2[1] - d1[1],
            )
            dist_p = math.hypot(
                self._physical_point2[0] - self._physical_point1[0],
                self._physical_point2[1] - self._physical_point1[1],
            )
            self._scale_row.set_subtitle("1.0 (locked)")
            if abs(dist_d - dist_p) > 0.1:
                self._warning_label.set_text(
                    _(
                        "Note: Scale is locked. Point 2 may "
                        "not align exactly if the physical "
                        "distance differs from the design "
                        "distance."
                    )
                )
                self._warning_label.set_visible(True)
            else:
                self._warning_label.set_visible(False)

    def _on_page_changed(self, stack, pspec):
        visible = stack.get_visible_child_name()
        if visible == "pick":
            self._left_panel.set_visible(True)
            self._back_btn.set_visible(False)
            self._reset_btn.set_visible(True)
            self._cancel_btn.set_visible(True)
            self._next_btn.set_visible(True)
            self._apply_btn.set_visible(False)
            self._update_pick_next_btn()
        elif visible == "jog":
            self._left_panel.set_visible(True)
            self._back_btn.set_visible(True)
            self._reset_btn.set_visible(False)
            self._cancel_btn.set_visible(True)
            self._next_btn.set_visible(True)
            self._apply_btn.set_visible(False)
            self._update_jog_next_btn()
        elif visible == "apply":
            self._left_panel.set_visible(True)
            self._back_btn.set_visible(True)
            self._reset_btn.set_visible(False)
            self._cancel_btn.set_visible(True)
            self._next_btn.set_visible(False)
            self._apply_btn.set_visible(True)
            self._apply_btn.set_sensitive(True)
            self._update_apply_preview()

    def _on_back_clicked(self, button):
        visible = self._right_stack.get_visible_child_name()
        if visible == "jog":
            self._right_stack.set_visible_child_name("pick")
        elif visible == "apply":
            self._right_stack.set_visible_child_name("jog")

    def _on_next_clicked(self, button):
        visible = self._right_stack.get_visible_child_name()
        if visible == "pick":
            self._right_stack.set_visible_child_name("jog")
        elif visible == "jog":
            self._right_stack.set_visible_child_name("apply")

    def _on_reset_clicked(self, button):
        self._pick_surface.reset()

    def _on_apply_clicked(self, button):
        if (
            self._design_point1 is None
            or self._design_point2 is None
            or self._physical_point1 is None
            or self._physical_point2 is None
        ):
            return

        d1, d2 = self._get_world_design_points()
        p1, p2 = self._get_world_physical_points()

        T = calculate_alignment_transform(
            d1, d2, p1, p2, allow_scale=self._allow_scale
        )

        old_matrix = self._workpiece.matrix.copy()
        new_matrix = T @ old_matrix

        self._editor.transform.create_transform_transaction(
            [(self._workpiece, old_matrix, new_matrix)]
        )

        toast = Adw.Toast(title=_("Alignment applied"))
        self.toast_overlay.add_toast(toast)
        self.close()

    def close(self):
        self._disable_focus()
        if self._machine:
            self._machine.state_changed.disconnect(
                self._on_machine_state_changed
            )
            self._machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            head = self._machine.get_default_head()
            head.changed.disconnect(self._on_head_changed)
        super().close()
