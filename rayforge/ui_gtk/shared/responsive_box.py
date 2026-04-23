from gi.repository import Gsk, Graphene, Gtk


class ResponsiveBox(Gtk.Widget):
    _SPACING = 12

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._first = None
        self._second = None

    def set_children(self, first, second=None):
        if self._first is not None:
            self._first.unparent()
        if self._second is not None:
            self._second.unparent()
        self._first = first
        self._second = second
        if first is not None:
            first.set_parent(self)
        if second is not None:
            second.set_parent(self)

    def do_get_request_mode(self):
        return Gtk.SizeRequestMode.WIDTH_FOR_HEIGHT

    def _get_vertical_height(self):
        assert self._first is not None
        assert self._second is not None
        h1 = self._first.measure(Gtk.Orientation.VERTICAL, -1)
        h2 = self._second.measure(Gtk.Orientation.VERTICAL, -1)
        return h1, h2

    def do_measure(self, orientation, for_size):
        if self._first is None:
            return (0, 0, -1, -1)
        if self._second is None:
            return self._first.measure(orientation, for_size)

        if orientation == Gtk.Orientation.VERTICAL:
            h1, h2 = self._get_vertical_height()
            stacked_nat = h1[1] + self._SPACING + h2[1]

            if for_size > 0:
                h1c = self._first.measure(Gtk.Orientation.VERTICAL, for_size)
                h2c = self._second.measure(Gtk.Orientation.VERTICAL, for_size)
                side_min = max(h1c[0], h2c[0])

                w1 = self._first.measure(Gtk.Orientation.HORIZONTAL, side_min)
                w2 = self._second.measure(Gtk.Orientation.HORIZONTAL, side_min)

                if for_size >= w1[0] + self._SPACING + w2[0]:
                    return (side_min, stacked_nat, -1, -1)

                return (
                    h1[0] + self._SPACING + h2[0],
                    stacked_nat,
                    -1,
                    -1,
                )

            return (max(h1[0], h2[0]), stacked_nat, -1, -1)

        if for_size > 0:
            h1, h2 = self._get_vertical_height()
            if for_size >= h1[1] + self._SPACING + h2[1]:
                w1 = self._first.measure(Gtk.Orientation.HORIZONTAL, h1[1])
                w2 = self._second.measure(Gtk.Orientation.HORIZONTAL, h2[1])
                return (max(w1[0], w2[0]), max(w1[1], w2[1]), -1, -1)

        w1 = self._first.measure(Gtk.Orientation.HORIZONTAL, for_size)
        w2 = self._second.measure(Gtk.Orientation.HORIZONTAL, for_size)
        return (
            w1[0] + self._SPACING + w2[0],
            w1[1] + self._SPACING + w2[1],
            -1,
            -1,
        )

    def do_size_allocate(self, width, height, baseline):
        if self._first is None:
            return
        if self._second is None:
            self._first.allocate(width, height, baseline, None)
            return

        h1 = self._first.measure(Gtk.Orientation.VERTICAL, -1)
        h2 = self._second.measure(Gtk.Orientation.VERTICAL, -1)

        if height >= h1[1] + self._SPACING + h2[1]:
            self._allocate_vertical(width, h1[1], h2[1])
        else:
            self._allocate_horizontal(width, height)

    def _allocate_vertical(self, width, h1_nat, h2_nat):
        assert self._first is not None
        assert self._second is not None
        w1 = self._first.measure(Gtk.Orientation.HORIZONTAL, h1_nat)
        w2 = self._second.measure(Gtk.Orientation.HORIZONTAL, h2_nat)
        child_w = min(max(w1[1], w2[1]), width)

        self._first.allocate(child_w, h1_nat, -1, None)

        transform = Gsk.Transform().translate(
            Graphene.Point().init(0, h1_nat + self._SPACING)
        )
        self._second.allocate(child_w, h2_nat, -1, transform)

    def _allocate_horizontal(self, width, height):
        assert self._first is not None
        assert self._second is not None
        w1 = self._first.measure(Gtk.Orientation.HORIZONTAL, height)
        w2 = self._second.measure(Gtk.Orientation.HORIZONTAL, height)

        available = width - self._SPACING
        total_nat = w1[1] + w2[1]

        if total_nat > 0 and available >= total_nat:
            w1w = w1[1]
            w2w = available - w1w
        elif total_nat > 0:
            ratio = available / total_nat
            w1w = max(w1[0], int(w1[1] * ratio))
            w2w = available - w1w
        else:
            w1w = available // 2
            w2w = available - w1w

        self._first.allocate(w1w, height, -1, None)

        transform = Gsk.Transform().translate(
            Graphene.Point().init(w1w + self._SPACING, 0)
        )
        self._second.allocate(w2w, height, -1, transform)
