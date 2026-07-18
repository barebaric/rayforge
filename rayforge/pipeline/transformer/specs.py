"""Batch transformer spec construction and Rust dispatch helpers.

This module owns the bridge between the Python transformer objects
(which remain the UI/settings/serialization layer) and the Rust
``Ops.apply_transformers()`` batch entry point.

The flow is:

1. ``build_transformer_specs()`` walks the enabled transformers and
   asks each one for its typed Rust ``*Spec`` via ``OpsTransformer.to_spec()``.
   ``to_spec()`` is total -- it must never return ``None`` -- so a
   misconfigured-but-enabled transformer raises here rather than
   silently producing wrong results.

2. ``apply_transformer_specs()`` hands the collected specs to
   ``Ops.apply_transformers()`` along with a small progress/cancel
   adapter that wraps the project's ``ProgressContext`` into the
   ``(progress, message)`` + ``is_cancelled()`` interface the Rust
   side expects.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
)

from raygeo.ops import Ops

from .base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from ...core.workpiece import WorkPiece
    from ...shared.tasker.progress import ProgressContext


def build_transformer_specs(
    transformers: Sequence[OpsTransformer],
    workpiece: Optional["WorkPiece"],
    stock_geometries: Optional[List["Geometry"]],
    settings: Optional[Dict[str, Any]],
) -> list:
    """Build typed Rust specs from enabled transformers.

    Iterates over ``transformers``, skipping disabled ones, and calls
    ``to_spec()`` on each enabled transformer. Because ``to_spec()``
    is total (never returns ``None``), there is no silent-skip branch
    here: a misconfigured-but-enabled transformer raises from
    ``to_spec()``.

    Args:
        transformers: The configured transformers (any phase).
        workpiece: The workpiece being processed, or None at step level.
        stock_geometries: Stock boundary geometries in world space.
        settings: Optional step settings dict.

    Returns:
        A list of typed ``*Spec`` pyclasses suitable for
        ``Ops.apply_transformers()``.
    """
    specs: list = []
    for t in transformers:
        if not t.enabled:
            continue
        specs.append(t.to_spec(workpiece, stock_geometries, settings))
    return specs


class _ProgressCallback:
    """Adapter exposing ``ProgressContext`` as the Rust progress interface.

    ``Ops.apply_transformers()`` expects ``progress_cb`` to be callable
    as ``cb(progress, message)`` and to expose ``is_cancelled()``.
    ``ProgressContext`` has ``set_progress`` / ``set_message`` /
    ``is_cancelled`` instead, so this adapter bridges the two.
    """

    def __init__(self, context: "ProgressContext") -> None:
        self._context = context

    def __call__(self, progress: float, message: str) -> None:
        self._context.set_progress(progress)
        if message:
            self._context.set_message(message)

    def is_cancelled(self) -> bool:
        return self._context.is_cancelled()


def apply_transformer_specs(
    ops: Ops,
    specs: list,
    context: Optional["ProgressContext"] = None,
) -> None:
    """Apply typed transformer specs via the Rust batch dispatch.

    Args:
        ops: The Ops object to transform in-place.
        specs: Typed ``*Spec`` pyclasses from ``build_transformer_specs``.
        context: Optional ProgressContext for progress reporting and
                 cancellation.
    """
    if not specs:
        return
    progress_cb = _ProgressCallback(context) if context else None
    ops.apply_transformers(specs, progress_cb=progress_cb)
