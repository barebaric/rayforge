from gi.repository import Gtk


css = """
button.round-button {
    min-width: 64px;
    min-height: 64px;
    border-radius: 32px;
    padding: 0;
    margin: 12px;
    background-color: @theme_selected_bg_color; /* Material primary color */
    color: @theme_selected_fg_color;
    font-size: 24px;
    border: none;
    box-shadow: 0 3px 6px rgba(0, 0, 0, 0.16),
                0 3px 6px rgba(0, 0, 0, 0.23); /* Shadow for depth */
    transition: background-color 0.2s, box-shadow 0.2s;
}

button.round-button:hover {
    background-color: shade(@theme_selected_bg_color, 0.9);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.19),
                0 6px 12px rgba(0, 0, 0, 0.23); /* Enhanced shadow on hover */
}

button.round-button:active {
    background-color: shade(@theme_selected_bg_color, 1.1); /* Lighter shade */
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.16),
                0 2px 4px rgba(0, 0, 0, 0.23); /* Reduced shadow on click */
}
"""


class RoundButton(Gtk.Button):
    def __init__(self, label, **kwargs):
        super().__init__(**kwargs)
        self.apply_css()
        self.set_label(label)
        self.set_halign(Gtk.Align.CENTER)

    def apply_css(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_string(css)
        style_context = self.get_style_context()
        style_context.add_provider(
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )
        style_context.add_class("round-button")
