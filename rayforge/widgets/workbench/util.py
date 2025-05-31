import logging
import cairo


logger = logging.getLogger(__name__)


def copy_surface(source, target, width, height, clip):
    in_width, in_height = source.get_width(), source.get_height()
    scale_x = width/in_width
    scale_y = height/in_height
    ctx = cairo.Context(target)
    # Apply clipping in the target context before scaling and painting
    if clip is not None:
        clip_x, clip_y, clip_w, clip_h = clip
        ctx.rectangle(clip_x, clip_y, clip_w, clip_h)
        ctx.clip()
    ctx.scale(scale_x, scale_y)
    ctx.set_source_surface(source, 0, 0)  # Set source surface at (0,0)
    ctx.paint()
    return target
