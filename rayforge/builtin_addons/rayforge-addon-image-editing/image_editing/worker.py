"""
Backend entry point for the image_editing addon.

Registers the image edit ops, which the pipeline needs in every process
that renders workpieces.
"""

from rayforge.core.hooks import hookimpl

from .ops import ALL_OPS

ADDON_NAME = "image_editing"


@hookimpl
def register_image_edits(image_edit_registry):
    """Register the image edit ops with the image edit registry."""
    for op_class in ALL_OPS:
        image_edit_registry.register(op_class, addon_name=ADDON_NAME)
