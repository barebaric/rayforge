from gi.repository import Gtk
from ..icons import get_icon
from .gtk import apply_css


css = """
.expander-card {
    background-color: @headerbar_bg_color;
    border-radius: 12px;
    box-shadow: 0 4px 10px alpha(black, 0.06);
    margin-bottom: 6px;
}

.expander-header {
    border-radius: 12px;
}

.expander-header:hover {
    background-color: shade(@headerbar_bg_color, 0.95);
}

.expander-card.expanded .expander-header {
    border-radius: 12px 12px 0 0;
    border-bottom: 1px solid @borders;
}

.expander-card.expanded .expander-header:hover {
    border-radius: 12px 12px 0 0;
}

.expander-title, .expander-subtitle {
    font-weight: normal;
}

.expander-subtitle {
    color: alpha(@headerbar_fg_color, 0.7);
}

.expander-arrow {
    transition: transform 0.2s ease-in-out, color 0.2s ease-in-out;
}

.expander-arrow.rotated {
    transform: rotate(90deg);
}

.expander-card>:nth-child(2) {
    border-radius: 0;
}
"""


class Expander(Gtk.Box):
    """
    A custom expander widget that looks and behaves like an Adwaita card.
    This version uses a Gtk.Box header and direct CSS class manipulation on
    the child icon to ensure reliable styling.
    """

    _css_loaded = False

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        apply_css(css)
        self.add_css_class("expander-card")

        # Header
        self.header = Gtk.Box()
        self.header.add_css_class("expander-header")
        self.append(self.header)

        # Add event controllers for click and hover
        click_controller = Gtk.GestureClick.new()
        click_controller.connect("released", self._on_header_clicked)
        self.header.add_controller(click_controller)

        header_content_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            margin_top=10,
            margin_bottom=10,
            margin_start=12,
            margin_end=12,
        )
        self.header.append(header_content_box)

        label_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        label_box.set_hexpand(True)
        header_content_box.append(label_box)

        self.title_label = Gtk.Label(xalign=0)
        self.title_label.add_css_class("expander-title")
        label_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(xalign=0)
        self.subtitle_label.add_css_class("expander-subtitle")
        label_box.append(self.subtitle_label)

        self.suffix_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header_content_box.append(self.suffix_box)

        self.arrow = get_icon("chevron-right-symbolic")
        self.arrow.add_css_class("expander-arrow")
        self.arrow.set_valign(Gtk.Align.CENTER)
        header_content_box.append(self.arrow)

        # Content revealer
        self.revealer = Gtk.Revealer()
        self.revealer.set_transition_type(
            Gtk.RevealerTransitionType.SLIDE_DOWN
        )
        self.revealer.connect("notify::reveal-child", self._update_state)
        self.append(self.revealer)

        self._update_state()

    def set_title(self, title: str):
        self.title_label.set_text(title)

    def set_subtitle(self, subtitle: str):
        self.subtitle_label.set_text(subtitle)

    def add_suffix(self, widget: Gtk.Widget):
        """Add a widget to the suffix area (between title and arrow)."""
        self.suffix_box.append(widget)

    def set_expanded(self, expanded: bool):
        self.revealer.set_reveal_child(expanded)

    def set_child(self, widget: Gtk.Widget):
        self.revealer.set_child(widget)

    def _on_header_clicked(self, *args):
        self.set_expanded(not self.revealer.get_reveal_child())

    def _update_state(self, *args):
        is_expanded = self.revealer.get_reveal_child()
        if is_expanded:
            # Add '.expanded' to the card for the header border
            self.add_css_class("expanded")
            # Add '.accent' and '.rotated' for color and rotation
            self.arrow.add_css_class("accent")
            self.arrow.add_css_class("rotated")
        else:
            self.remove_css_class("expanded")
            self.arrow.remove_css_class("accent")
            self.arrow.remove_css_class("rotated")


class ExpanderWithButton(Expander):
    """
    An Expander that includes a "+" icon button in the header suffix area
    and a content box for child widgets.
    """

    def __init__(self, button_label: str, **kwargs):
        super().__init__(**kwargs)
        self.content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_child(self.content_box)

        self.add_button = Gtk.Button()
        self.add_button.set_tooltip_text(button_label)
        self.add_button.set_child(get_icon("add-symbolic"))
        self.add_suffix(self.add_button)

    def append_content(self, widget: Gtk.Widget):
        self.content_box.append(widget)
