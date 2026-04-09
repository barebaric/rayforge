import logging
from gi.repository import Gsk, Graphene, Gtk, Gdk, GLib
from blinker import Signal
from .dock_area import DockArea
from .gtk import apply_css

logger = logging.getLogger(__name__)

divider_css = """
.dock-divider {
    background: transparent;
    transition: background 100ms ease;
}
.dock-divider:hover {
    background: alpha(@theme_fg_color, 0.08);
}
.dock-divider.drop-highlight {
    background: alpha(@theme_selected_bg_color, 0.25);
    border-left: 2px solid @theme_selected_bg_color;
    border-right: 2px solid @theme_selected_bg_color;
}
"""

apply_css(divider_css)


class DockLayout(Gtk.Widget):
    _DIVIDER_WIDTH = 6

    def __init__(self, orientation=Gtk.Orientation.HORIZONTAL, **kwargs):
        super().__init__(**kwargs)
        self.set_hexpand(True)
        self.set_vexpand(True)

        self._items = {}
        self._areas = []
        self._dividers = []
        self._orientation = orientation
        self._area_sizes = []
        self._drag_start_sizes = []
        self._dragging = False
        self._dragging_divider = -1
        self._drag_start_x = 0

        self.layout_changed = Signal()
        self.tab_changed = Signal()

        gesture = Gtk.GestureDrag()
        gesture.connect("drag-begin", self._on_drag_begin)
        gesture.connect("drag-update", self._on_drag_update)
        gesture.connect("drag-end", self._on_drag_end)
        self.add_controller(gesture)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        self.add_controller(motion)

    def register_item(self, item):
        self._items[item.name] = item

    def get_item(self, name):
        return self._items.get(name)

    def get_item_names(self):
        return list(self._items.keys())

    def add_area(self, position=-1):
        area = DockArea(orientation=self._orientation)
        area.layout_changed.connect(self._on_area_layout_changed)
        area.tab_changed.connect(self._on_area_tab_changed)
        area.item_dropped.connect(self._on_area_item_dropped)

        if position < 0 or position >= len(self._areas):
            self._areas.append(area)
        else:
            self._areas.insert(position, area)
        self._rebuild_layout()
        return area

    def remove_area(self, area):
        if area in self._areas:
            self._areas.remove(area)
            self._rebuild_layout()

    def get_areas(self):
        return list(self._areas)

    def get_area_count(self):
        return len(self._areas)

    def find_item_area(self, name):
        for area in self._areas:
            if area.has_item(name):
                return area
        return None

    def move_item(self, item_name, to_area, position=-1):
        source_area = self.find_item_area(item_name)
        if source_area is None:
            return
        if source_area is to_area:
            return
        item = source_area.remove_item(item_name)
        if item is None:
            return
        to_area.add_item(item, position)
        if source_area.item_count() == 0:
            self.remove_area(source_area)
        self.layout_changed.send(self)

    def _on_area_layout_changed(self, sender):
        self.layout_changed.send(self)

    def _on_area_tab_changed(self, sender, *, name):
        self.tab_changed.send(self, name=name)

    def _on_area_item_dropped(self, sender, *, name):
        target_area = sender
        GLib.idle_add(self._deferred_move_item, name, target_area)

    def _deferred_move_item(self, item_name, to_area):
        DockArea._drag_source_area = None
        self.move_item(item_name, to_area)
        return GLib.SOURCE_REMOVE

    def _rebuild_layout(self):
        child = self.get_first_child()
        while child is not None:
            next_child = child.get_next_sibling()
            child.unparent()
            child = next_child

        self._dividers.clear()
        self._area_sizes = []

        if not self._areas:
            return

        for area in self._areas:
            area.set_parent(self)

        for i in range(len(self._areas) - 1):
            divider = self._create_divider(i)
            divider.set_parent(self)
            self._dividers.append(divider)

    def _create_divider(self, index):
        divider = Gtk.Box()
        divider.add_css_class("dock-divider")

        drop = Gtk.DropTarget.new(str, Gdk.DragAction.MOVE)
        drop.connect("enter", self._on_divider_drop_enter, divider)
        drop.connect("leave", self._on_divider_drop_leave, divider)
        drop.connect("drop", self._on_divider_drop, index)
        divider.add_controller(drop)

        return divider

    def do_get_request_mode(self):
        return Gtk.SizeRequestMode.HEIGHT_FOR_WIDTH

    def do_measure(self, orientation, for_size):
        if not self._areas:
            return (0, 0, -1, -1)

        if orientation == Gtk.Orientation.VERTICAL:
            max_min = 0
            max_nat = 0
            for area in self._areas:
                m, n, _, _ = area.measure(orientation, for_size)
                max_min = max(max_min, m)
                max_nat = max(max_nat, n)
            return (max_min, max_nat, -1, -1)

        total_min = 0
        total_nat = 0
        for area in self._areas:
            m, n, _, _ = area.measure(orientation, for_size)
            total_min += m
            total_nat += n
        dividers = max(0, len(self._areas) - 1) * self._DIVIDER_WIDTH
        return (total_min + dividers, total_nat + dividers, -1, -1)

    def do_size_allocate(self, width, height, baseline):
        if not self._areas:
            return

        n = len(self._areas)
        divider_total = max(0, n - 1) * self._DIVIDER_WIDTH
        available = width - divider_total

        if not self._area_sizes or len(self._area_sizes) != n:
            self._compute_default_sizes(available, height)

        if self._dragging:
            sizes = list(self._area_sizes)
        else:
            sizes = self._fit_sizes(available, height)

        x = 0
        for i, area in enumerate(self._areas):
            transform = Gsk.Transform().translate(Graphene.Point().init(x, 0))
            area.allocate(sizes[i], height, baseline, transform)
            x += sizes[i]

            if i < len(self._dividers):
                div_transform = Gsk.Transform().translate(
                    Graphene.Point().init(x, 0)
                )
                self._dividers[i].allocate(
                    self._DIVIDER_WIDTH, height, baseline, div_transform
                )
                x += self._DIVIDER_WIDTH

        self._area_sizes = sizes

    def _compute_default_sizes(self, available, height):
        n = len(self._areas)
        sizes = [0] * n

        remaining = available
        for i, area in enumerate(self._areas):
            if not area.get_hexpand():
                _, nat, _, _ = area.measure(Gtk.Orientation.HORIZONTAL, height)
                sizes[i] = nat
                remaining -= nat

        expandable = [i for i, a in enumerate(self._areas) if a.get_hexpand()]
        if expandable:
            share = max(50, remaining // len(expandable))
            for i in expandable:
                sizes[i] = share

        diff = available - sum(sizes)
        if expandable:
            sizes[expandable[-1]] += diff

        self._area_sizes = sizes

    def _fit_sizes(self, available, height):
        sizes = list(self._area_sizes)
        n = len(sizes)
        if n == 0:
            return sizes

        for i in range(n):
            if not self._areas[i].get_hexpand():
                _, nat, _, _ = self._areas[i].measure(
                    Gtk.Orientation.HORIZONTAL, height
                )
                sizes[i] = nat

        fixed_total = sum(
            sizes[i] for i in range(n) if not self._areas[i].get_hexpand()
        )
        remaining = available - fixed_total
        expandable = [i for i in range(n) if self._areas[i].get_hexpand()]

        if expandable:
            exp_total = sum(sizes[i] for i in expandable)
            if exp_total > 0:
                ratio = remaining / exp_total
                for i in expandable:
                    sizes[i] = max(50, int(sizes[i] * ratio))
            else:
                share = remaining // len(expandable)
                for i in expandable:
                    sizes[i] = max(50, share)

            diff = available - sum(sizes)
            if expandable:
                sizes[expandable[-1]] += diff

        return sizes

    def _divider_at_x(self, x):
        if len(self._area_sizes) < 2:
            return -1

        pos = 0
        for i in range(len(self._area_sizes)):
            pos += self._area_sizes[i]
            if abs(x - (pos + self._DIVIDER_WIDTH / 2)) <= self._DIVIDER_WIDTH:
                if i + 1 < len(self._areas):
                    left = self._areas[i].get_hexpand()
                    right = self._areas[i + 1].get_hexpand()
                    if left and right:
                        return i
            pos += self._DIVIDER_WIDTH

        return -1

    def _on_drag_begin(self, gesture, x, y):
        div = self._divider_at_x(x)
        if div < 0:
            gesture.set_state(Gtk.EventSequenceState.DENIED)
            return
        self._dragging_divider = div
        self._drag_start_x = x
        self._drag_start_sizes = list(self._area_sizes)
        self._dragging = True
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _on_drag_update(self, gesture, offset_x, offset_y):
        if not self._dragging:
            return

        i = self._dragging_divider
        delta = int(offset_x)
        sizes = list(self._drag_start_sizes)

        new_left = sizes[i] + delta
        new_right = sizes[i + 1] - delta

        alloc = self.get_allocation()
        height = alloc.height
        left_min, _, _, _ = self._areas[i].measure(
            Gtk.Orientation.HORIZONTAL, height
        )
        right_min, _, _, _ = self._areas[i + 1].measure(
            Gtk.Orientation.HORIZONTAL, height
        )

        if new_left < left_min:
            new_left = left_min
            new_right = sizes[i] + sizes[i + 1] - new_left
        if new_right < right_min:
            new_right = right_min
            new_left = sizes[i] + sizes[i + 1] - new_right

        self._area_sizes[i] = new_left
        self._area_sizes[i + 1] = new_right
        self.queue_allocate()

    def _on_drag_end(self, gesture, x, y):
        if self._dragging:
            self.layout_changed.send(self)
        self._dragging = False
        self._dragging_divider = -1
        self._drag_start_sizes = []

    def _on_motion(self, ctrl, x, y):
        if self._divider_at_x(x) >= 0:
            self.set_cursor_from_name("col-resize")
        else:
            self.set_cursor(None)

    def _on_divider_drop_enter(self, target, x, y, divider):
        divider.add_css_class("drop-highlight")
        return Gdk.DragAction.MOVE

    def _on_divider_drop_leave(self, target, divider):
        divider.remove_css_class("drop-highlight")

    def _on_divider_drop(self, target, value, x, y, index):
        divider = target.get_widget()
        divider.remove_css_class("drop-highlight")
        if not isinstance(value, str):
            return False

        item_name = value
        source_area = self.find_item_area(item_name)
        if source_area is None:
            return False

        if source_area.item_count() == 1 and source_area in self._areas:
            source_idx = self._areas.index(source_area)
            if index == source_idx or index == source_idx - 1:
                return False

        insert_idx = index + 1
        GLib.idle_add(self._deferred_edge_drop, item_name, insert_idx)
        return True

    def _deferred_edge_drop(self, item_name, insert_idx):
        DockArea._drag_source_area = None
        source_area = self.find_item_area(item_name)
        if source_area is None:
            return GLib.SOURCE_REMOVE

        new_area = self.add_area(insert_idx)
        self.move_item(item_name, new_area)
        self.layout_changed.send(self)
        return GLib.SOURCE_REMOVE

    def get_layout(self):
        areas = []
        for i, area in enumerate(self._areas):
            area_data = area.get_layout()
            if i < len(self._area_sizes):
                area_data["size"] = self._area_sizes[i]
            areas.append(area_data)
        return {"areas": areas}

    def apply_layout(self, layout):
        areas_data = layout.get("areas", [])
        if not areas_data:
            return

        all_item_names = list(self._items.keys())
        placed = set()
        saved_sizes = []

        new_areas = []
        for area_data in areas_data:
            items = area_data.get("items", [])
            valid = [n for n in items if n in self._items]
            if not valid:
                continue

            area = DockArea(orientation=self._orientation)
            area.layout_changed.connect(self._on_area_layout_changed)
            area.tab_changed.connect(self._on_area_tab_changed)
            area.item_dropped.connect(self._on_area_item_dropped)
            for name in valid:
                area.add_item(self._items[name])
                placed.add(name)
            active = area_data.get("active")
            if active and active in area.items:
                area.set_active_item(active)
            new_areas.append(area)
            size = area_data.get("size")
            saved_sizes.append(size if isinstance(size, int) else None)

        unplaced = [n for n in all_item_names if n not in placed]
        if unplaced and new_areas:
            for name in unplaced:
                new_areas[0].add_item(self._items[name])
            saved_sizes.append(None)
        elif unplaced:
            area = DockArea(orientation=self._orientation)
            area.layout_changed.connect(self._on_area_layout_changed)
            area.tab_changed.connect(self._on_area_tab_changed)
            area.item_dropped.connect(self._on_area_item_dropped)
            for name in unplaced:
                area.add_item(self._items[name])
            new_areas.append(area)
            saved_sizes.append(None)

        for old_area in self._areas:
            for name in list(old_area.get_item_names()):
                old_area.remove_item(name)

        self._areas = new_areas
        self._rebuild_layout()

        if len(saved_sizes) == len(self._areas) and all(
            s is not None for s in saved_sizes
        ):
            self._area_sizes = saved_sizes
            self.queue_allocate()

    def set_area_active(self, area_index, item_name):
        if 0 <= area_index < len(self._areas):
            self._areas[area_index].set_active_item(item_name)
