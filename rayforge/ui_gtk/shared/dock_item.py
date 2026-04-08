from .gtk import apply_css


dock_css = """
box.dock-edge-zone {
    min-width: 6px;
    min-height: 6px;
    background: transparent;
    transition: background 150ms ease;
}

box.dock-edge-zone.highlight-left {
    background: alpha(@theme_selected_bg_color, 0.3);
    border-left: 2px solid @theme_selected_bg_color;
}

box.dock-edge-zone.highlight-right {
    background: alpha(@theme_selected_bg_color, 0.3);
    border-right: 2px solid @theme_selected_bg_color;
}

box.dock-area-drop-highlight {
    background: alpha(@theme_selected_bg_color, 0.08);
}
"""


apply_css(dock_css)


class DockItem:
    def __init__(self, name, icon_name, widget, label=None, expands=True):
        self.name = name
        self.icon_name = icon_name
        self.widget = widget
        self.label = label or name
        self.expands = expands
