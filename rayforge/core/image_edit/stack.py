from __future__ import annotations

import logging
from collections.abc import Iterable

import pyvips

from .base import ImageEditStep
from .registry import ImageEditRegistry, image_edit_registry

logger = logging.getLogger(__name__)


def apply_image_edits(
    image: pyvips.Image,
    edits: Iterable[ImageEditStep],
    registry: ImageEditRegistry = image_edit_registry,
) -> pyvips.Image:
    """
    Applies the enabled edits in order. An op that is not registered
    (its addon is missing or disabled) is skipped, so the document stays
    usable and keeps the edit for when the op is available again.
    """
    for edit in edits:
        if not edit.enabled:
            continue
        op_class = registry.get(edit.op)
        if op_class is None:
            logger.debug(f"Skipping unregistered image edit '{edit.op}'")
            continue
        image = op_class.apply(image, edit.params)
    return image


def missing_image_edit_ops(
    edits: Iterable[ImageEditStep],
    registry: ImageEditRegistry = image_edit_registry,
) -> set[str]:
    """Names of the ops in ``edits`` that are not registered."""
    return {edit.op for edit in edits if registry.get(edit.op) is None}
