import cairo
import numpy as np
from .modifier import Modifier


def make_transparent_except_color(surface, target_r, target_g, target_b):
    if surface.get_format() != cairo.FORMAT_ARGB32:
        raise ValueError("Surface must be in ARGB32 format.")

    width, height = surface.get_width(), surface.get_height()
    stride = surface.get_stride()

    # Get pixel data as a NumPy array
    data = surface.get_data()
    buf = np.frombuffer(data, dtype=np.uint8).reshape((height, stride))

    # Convert to 32-bit ARGB view
    argb = buf.view(dtype=np.uint32)[:, :width]

    # Extract color channels
    r = (argb >> 16) & 0xFF
    g = (argb >> 8) & 0xFF
    b_channel = argb & 0xFF

    # Create mask for pixels not matching the target color
    mask = ~((r == target_r) & (g == target_g) & (b_channel == target_b))

    # Set alpha to 0 for non-matching pixels
    argb[mask] = ((0x00 << 24) | (r[mask] << 16) |
                  (g[mask] << 8) | b_channel[mask])


class KeepColor(Modifier):
    """
    Makes everything except for a selected RGB color transparent.
    """
    def __init__(self, r, g, b):
        super().__init__()
        self.color = r, g, b

    def to_dict(self):
        d = super().to_dict()
        d.update({
            'r': self.color[0],
            'g': self.color[1],
            'b': self.color[2],
        })
        return d

    @classmethod
    def from_dict(cls, data: dict):
        return cls(r=data['r'], g=data['g'], b=data['b'])

    def run(self, surface):
        make_transparent_except_color(surface, *self.color)
