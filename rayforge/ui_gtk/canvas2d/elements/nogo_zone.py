import math

import cairo

from ....machine.models.zone import Zone, ZoneShape
from ...canvas import CanvasElement

_FILL_COLOR = (1.0, 0.2, 0.2, 0.15)
_STROKE_COLOR = (1.0, 0.0, 0.0, 0.6)
_HATCH_COLOR = (1.0, 0.0, 0.0, 0.25)
_HATCH_SPACING = 4.0


class NogoZoneElement(CanvasElement):
    """
    A non-interactive CanvasElement that draws a no-go zone as a
    semi-transparent red filled rectangle with a border and diagonal
    hatch lines. Zones with BOX or CYLINDER shape are projected to
    their X/Y footprint in the 2D view.
    """

    def __init__(self, zone: Zone, **kwargs):
        super().__init__(
            x=0,
            y=0,
            width=10.0,
            height=10.0,
            selectable=False,
            draggable=False,
            clip=False,
            data=zone,
            **kwargs,
        )
        self._fill_color = _FILL_COLOR
        self._stroke_color = _STROKE_COLOR
        self._hatch_color = _HATCH_COLOR
        self._update_from_zone()

    def _update_from_zone(self):
        zone: Zone = self.data
        p = zone.params
        x = p.get("x", 0.0)
        y = p.get("y", 0.0)

        if zone.shape == ZoneShape.CYLINDER:
            r = p.get("radius", 5.0)
            w = r * 2
            h = r * 2
            x -= r
            y -= r
        else:
            w = p.get("w", 10.0)
            h = p.get("h", 10.0)

        self.set_pos(x, y)
        self.set_size(w, h)
        self.set_visible(zone.enabled)

    def draw(self, ctx: cairo.Context):
        zone: Zone = self.data

        ctx.save()
        ctx.set_source_rgba(*self._fill_color)
        if zone.shape == ZoneShape.CYLINDER:
            r = self.width / 2.0
            ctx.arc(r, r, r, 0, 2 * math.pi)
        else:
            ctx.rectangle(0, 0, self.width, self.height)
        ctx.fill_preserve()

        ctx.set_source_rgba(*self._stroke_color)
        ctx.set_hairline(True)
        ctx.stroke()

        if zone.shape == ZoneShape.CYLINDER:
            r = self.width / 2.0
            ctx.arc(r, r, r, 0, 2 * math.pi)
        else:
            ctx.rectangle(0, 0, self.width, self.height)
        ctx.clip()
        self._draw_hatch(ctx)
        ctx.restore()

    def _draw_hatch(self, ctx: cairo.Context):
        ctx.set_source_rgba(*self._hatch_color)
        ctx.set_hairline(True)

        w, h = self.width, self.height
        step = _HATCH_SPACING

        for offset in _frange(-w - h, w + h, step):
            ctx.move_to(offset, 0)
            ctx.line_to(offset + h, h)

        ctx.stroke()


def _frange(start: float, stop: float, step: float):
    vals = []
    v = start
    while v < stop:
        vals.append(v)
        v += step
    return vals
