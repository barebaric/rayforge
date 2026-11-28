"""The base row widget shared by all tree node types."""

from gettext import gettext as _

from gi.repository import Gdk, GLib, Gtk, Pango

from ...ui_gtk.icons import get_icon
from ...ui_gtk.shared.gtk import apply_css
from .specs import NodeSpec

_CSS = """
.rf-tree-list > row {
    background-color: transparent;
    border-radius: 6px;
    margin: 0 4px;
    padding: 0;
    border: none;
}
.rf-tree-list > row .rf-row {
    padding: 1px 4px;
    border-radius: 6px;
}
.rf-tree-list > row:hover .rf-row {
    background-color: alpha(currentColor, 0.06);
}
.rf-tree-list > row.selected-row .rf-row {
    background-color: alpha(@accent_bg_color, 0.25);
}
.rf-tree-list > row.off .rf-name,
.rf-tree-list > row.off .rf-detail,
.rf-tree-list > row.off .rf-icon {
    opacity: 0.45;
}
.rf-detail {
    font-size: 93%;
    opacity: 0.6;
}
.rf-duration {
    font-size: 93%;
    opacity: 0.6;
    font-variant-numeric: tabular-nums;
}
.rf-badge {
    min-width: 16px;
    min-height: 16px;
    border-radius: 10px;
    background-color: @accent_bg_color;
    color: @accent_fg_color;
    font-size: 80%;
    font-weight: 700;
}
.rf-pill {
    background-color: alpha(currentColor, 0.08);
    border-radius: 99px;
    padding: 5px 12px;
}
.rf-pill-title {
    font-weight: 700;
}
.rf-pill-swatch {
    padding: 0;
    min-width: 22px;
    min-height: 22px;
    border-radius: 8px;
}
.rf-swatch {
    min-width: 22px;
    min-height: 22px;
    border-radius: 6px;
    padding: 0;
}
.rf-swatch.taken {
    box-shadow: 0 0 0 2px @accent_bg_color;
}
"""


def _hex_to_rgb(color: str) -> tuple[float, float, float]:
    value = color.lstrip("#")
    return (
        int(value[0:2], 16) / 255,
        int(value[2:4], 16) / 255,
        int(value[4:6], 16) / 255,
    )


def _draw_circle(area, ctx, width, height, color: str):
    ctx.set_source_rgb(*_hex_to_rgb(color))
    ctx.arc(width / 2, height / 2, min(width, height) / 2, 0, 6.2832)
    ctx.fill()


def _draw_rect(area, ctx, width, height, color: str):
    radius = min(width, height) * 0.25
    ctx.set_source_rgb(*_hex_to_rgb(color))
    ctx.new_sub_path()
    ctx.arc(width - radius, radius, radius, -90 * 3.1416 / 180, 0)
    ctx.arc(width - radius, height - radius, radius, 0, 90 * 3.1416 / 180)
    ctx.arc(
        radius,
        height - radius,
        radius,
        90 * 3.1416 / 180,
        180 * 3.1416 / 180,
    )
    ctx.arc(radius, radius, radius, 180 * 3.1416 / 180, 270 * 3.1416 / 180)
    ctx.close_path()
    ctx.fill()


def color_widget(color: str) -> Gtk.Widget:
    """Returns a small round dot filled with the hex color."""
    area = Gtk.DrawingArea()
    area.set_size_request(11, 11)
    area.set_valign(Gtk.Align.CENTER)
    area.set_draw_func(_draw_circle, color)
    return area


def swatch_button(color: str, diameter: int = 22) -> Gtk.Button:
    """Returns a flat button filled edge to edge with the hex color."""
    button = Gtk.Button(css_classes=["rf-swatch"])
    area = Gtk.DrawingArea()
    area.set_size_request(diameter, diameter)
    area.set_draw_func(_draw_rect, color)
    button.set_child(area)
    return button


class NodeRow(Gtk.ListBoxRow):
    """Base class for all node rows.

    Subclasses build their :class:`NodeSpec` from a model item and pass
    it, together with the owning tab (``callbacks``), to this base. The
    callbacks object provides ``on_row_activated(spec)``,
    ``on_eye_toggled(spec)``, ``on_twisty_toggled(spec)`` and
    ``on_row_renamed(spec, name)``.
    """

    def __init__(self, spec: NodeSpec, callbacks):
        super().__init__()
        apply_css(_CSS)
        self.spec = spec
        self._callbacks = callbacks

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.add_css_class("rf-row")
        box.set_margin_start(6 + spec.depth * 16)
        box.set_margin_end(6)
        self.set_child(box)
        if spec.selected:
            self.add_css_class("selected-row")
        if not spec.on:
            self.add_css_class("off")

        box.append(self._build_twisty())
        box.append(self._build_lead())
        self._name_label = self._build_name()
        box.append(self._name_label)
        box.append(self._build_detail())
        box.append(self._build_duration())
        box.append(self._build_eye())

        click = Gtk.GestureClick()
        click.set_button(1)
        click.connect("released", self._on_clicked)
        self.add_controller(click)

    def _on_clicked(self, gesture, n_press, x, y):
        if not self.spec.selectable:
            return
        self._callbacks.on_row_activated(self.spec)

    def _build_twisty(self) -> Gtk.Widget:
        button = Gtk.Button()
        button.add_css_class("flat")
        button.set_valign(Gtk.Align.CENTER)
        if self.spec.has_children:
            icon = (
                "go-down-symbolic"
                if self.spec.expanded
                else "chevron-right-symbolic"
            )
            image = get_icon(icon)
            image.set_pixel_size(12)
            button.set_child(image)
            button.connect(
                "clicked",
                lambda b: self._callbacks.on_twisty_toggled(self.spec),
            )
        else:
            button.set_sensitive(False)
            button.set_opacity(0)
        button.set_size_request(20, 20)
        return button

    def _build_lead(self) -> Gtk.Widget:
        lead = Gtk.Box()
        lead.set_size_request(16, 16)
        lead.set_valign(Gtk.Align.CENTER)
        if self.spec.color:
            lead.append(color_widget(self.spec.color))
        elif self.spec.badge:
            badge = Gtk.Label(label=self.spec.badge)
            badge.add_css_class("rf-badge")
            badge.set_valign(Gtk.Align.CENTER)
            lead.append(badge)
        elif self.spec.icon:
            image = get_icon(self.spec.icon)
            image.set_pixel_size(14)
            image.add_css_class("rf-icon")
            image.set_valign(Gtk.Align.CENTER)
            image.set_opacity(0.8)
            lead.append(image)
        return lead

    def _build_name(self) -> Gtk.Widget:
        name = Gtk.Label(label=self.spec.label, xalign=0)
        name.add_css_class("rf-name")
        name.set_ellipsize(Pango.EllipsizeMode.END)
        name.set_hexpand(True)
        return name

    def start_rename(self):
        """Replaces the name label with an inline entry.

        Enter or leaving the entry commits the new name through the
        owning tab; Escape cancels. The entry replaces the label in
        place, so the row keeps its layout while editing.
        """
        if not self.spec.selectable:
            return
        parent = self._name_label.get_parent()
        if not isinstance(parent, Gtk.Box):
            return
        entry = Gtk.Entry()
        entry.add_css_class("rf-rename")
        entry.set_text(self.spec.label)
        entry.set_hexpand(True)
        entry.set_valign(Gtk.Align.CENTER)

        committed = False

        def focus_entry(*_args):
            entry.grab_focus()
            entry.select_region(0, -1)

        def finish(rename: bool) -> bool:
            name = entry.get_text().strip()
            entry.unparent()
            self._name_label.set_visible(True)
            if rename and name and name != self.spec.label:
                self._callbacks.on_row_renamed(self.spec, name)
            return GLib.SOURCE_REMOVE

        def commit(*_args):
            nonlocal committed
            if committed:
                return
            committed = True
            # The focus change that triggered this still manipulates the
            # entry; removing it right away corrupts GTK's focus
            # bookkeeping, so tear down once that change is complete.
            GLib.timeout_add(0, finish, True)

        def cancel(*_args):
            nonlocal committed
            if committed:
                return
            committed = True
            GLib.timeout_add(0, finish, False)

        entry.connect("activate", commit)
        focus = Gtk.EventControllerFocus()
        focus.connect("leave", lambda _c: commit())
        entry.add_controller(focus)

        def on_key_pressed(_controller, keyval, *_args):
            if keyval == Gdk.KEY_Escape:
                cancel()
                return True
            return False

        keys = Gtk.EventControllerKey()
        keys.connect("key-pressed", on_key_pressed)
        entry.add_controller(keys)

        # Inserting into a mapped parent maps the entry right away, so
        # the handlers must be in place before that.
        entry.connect("map", focus_entry)
        parent.insert_child_after(entry, self._name_label)
        self._name_label.set_visible(False)

    def _build_detail(self) -> Gtk.Widget:
        detail = Gtk.Label(label=self.spec.detail, xalign=1)
        detail.add_css_class("rf-detail")
        detail.set_ellipsize(Pango.EllipsizeMode.END)
        detail.set_max_width_chars(24)
        detail.set_visible(bool(self.spec.detail))
        return detail

    def _build_duration(self) -> Gtk.Widget:
        duration = Gtk.Label(label=self.spec.duration, xalign=1)
        duration.add_css_class("rf-duration")
        duration.set_visible(bool(self.spec.duration))
        return duration

    def _build_eye(self) -> Gtk.Widget:
        button = Gtk.Button()
        button.add_css_class("flat")
        button.set_valign(Gtk.Align.CENTER)
        button.set_size_request(22, 22)
        if not self.spec.eye:
            button.set_visible(False)
            button.set_sensitive(False)
            return button
        icon = (
            "visibility-on-symbolic"
            if self.spec.on
            else "visibility-off-symbolic"
        )
        image = get_icon(icon)
        image.set_pixel_size(14)
        button.set_child(image)
        button.set_tooltip_text(_("Hide") if self.spec.on else _("Show"))
        button.connect(
            "clicked", lambda b: self._callbacks.on_eye_toggled(self.spec)
        )
        return button
