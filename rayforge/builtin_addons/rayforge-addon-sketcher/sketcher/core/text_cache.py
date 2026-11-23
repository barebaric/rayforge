"""Text-template resolution and geometry caching for sketch documents.

Raygeo evaluates expressions and resolves the individual placeholders,
including their dependency stamps; this module connects that wasm-safe
Rust machinery to the sketch document and caches the generated geometry.

A cached geometry result is reused only when the values of the names the
sketch reads and the resolved text entries are unchanged, which keeps
unrelated document changes cheap without ever serving stale text.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from raygeo.geo import Geometry
from raygeo.sketch.document import FillStyle
from raygeo.sketch.entities import TextBox
from raygeo.sketch.params import ParameterContext
from raygeo.template import resolve_text as resolve_template

from rayforge.core.expression import (
    names_in_expression,
    referenced_template_names,
)
from rayforge.image.structures import FillRenderData

from ..template_functions import get_template_functions

DEFAULT_FILL_COLOR = (0.85, 0.85, 0.85, 0.7)
GEOMETRY_CACHE_SIZE = 8


def _copy_fill(fill: FillRenderData) -> FillRenderData:
    """Copies a fill deep enough for callers to transform it."""
    return FillRenderData(
        geometry=fill.geometry.copy(),
        style=fill.style,
        color=fill.color,
        gradient_stops=(
            list(fill.gradient_stops)
            if fill.gradient_stops is not None
            else None
        ),
        gradient_angle=fill.gradient_angle,
    )


class GeometryCache:
    """A small least-recently-used cache of rendered sketch geometry."""

    def __init__(self, max_size: int = GEOMETRY_CACHE_SIZE):
        self._max_size = max_size
        self._entries: OrderedDict[Any, tuple] = OrderedDict()

    def get(self, key: Any) -> tuple | None:
        """Returns the cached entry, marking it as recently used."""
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(
        self,
        key: Any,
        geometry: Geometry,
        fills: list[FillRenderData],
        entries: dict,
    ) -> None:
        """Caches a result, evicting the oldest entry when full."""
        self._entries[key] = (
            geometry.copy(),
            [_copy_fill(fill) for fill in fills],
            dict(entries),
        )
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Drops every cached result."""
        self._entries.clear()


def template_sources(document: Any) -> list[str]:
    """The raw contents of all text boxes in the document."""
    return [
        entity.content for entity in _text_boxes(document) if entity.content
    ]


def variables_used(document: Any) -> set[str]:
    """The template variable names the document's text boxes read."""
    return referenced_template_names(template_sources(document))


def resolve_text(
    document: Any,
    *,
    text_values: dict[str, Any] | None = None,
    resolved_text_cache: dict | None = None,
) -> dict[int, str | None]:
    """Resolves the document's text templates without building geometry."""
    context = _text_context(document, text_values)
    entries = resolved_text_cache if resolved_text_cache is not None else {}
    resolved, _volatile = _resolve_text(document, context, entries)
    return resolved


def render(
    document: Any,
    params: dict[str, Any] | None = None,
    *,
    initial_values: dict[str, Any] | None = None,
    text_values: dict[str, Any] | None = None,
    resolved_text_cache: dict | None = None,
    cache: GeometryCache | None = None,
) -> tuple[Geometry, list[FillRenderData]]:
    """Generates geometry and fills for the document.

    ``params`` override the document's parameters for this request.
    ``initial_values`` seed the parameter evaluation (the host's input
    parameter values). ``text_values`` are extra template-only values,
    such as a job counter, and are not part of the solve. The mutable
    ``resolved_text_cache`` carries resolved text across calls so
    volatile expressions stay stable for one caller.
    """
    cache_entries = (
        resolved_text_cache if resolved_text_cache is not None else {}
    )
    function_names = set(get_template_functions())
    read_names = _read_names(document, initial_values)
    values = _effective_values(document, params, initial_values, text_values)
    value_key = _value_key(values, read_names, function_names)
    seed_key = _text_key(cache_entries)
    lookup_key = (value_key, seed_key)

    if cache is not None:
        hit = cache.get(lookup_key)
        if hit is not None:
            geometry, fills, entries = hit
            cache_entries.update(entries)
            return geometry.copy(), [_copy_fill(fill) for fill in fills]

    clone = document.clone_for_geometry()
    clone.solve(
        variable_overrides=params,
        initial_values=initial_values,
        template_functions=get_template_functions(),
    )
    context = dict(clone.solved_ctx or {})
    context.update(text_values or {})

    resolved, volatile = _resolve_text(clone, context, cache_entries)
    geometry = clone.to_geometry(resolved)
    fills = _build_fills(clone, resolved)

    if cache is not None:
        entries = dict(cache_entries)
        cache.put((value_key, _text_key(entries)), geometry, fills, entries)
        if not volatile:
            cache.put(lookup_key, geometry, fills, entries)

    return geometry, fills


def to_geometry(
    document: Any,
    *,
    text_values: dict[str, Any] | None = None,
    resolved_text_cache: dict | None = None,
) -> Geometry:
    """Returns the document's geometry with text templates resolved."""
    context = _text_context(document, text_values)
    entries = resolved_text_cache if resolved_text_cache is not None else {}
    resolved, _volatile = _resolve_text(document, context, entries)
    return document.to_geometry(resolved)


def get_fill_render_data(
    document: Any,
    *,
    text_values: dict[str, Any] | None = None,
    resolved_text_cache: dict | None = None,
    exclude_ids: Iterable[int] | None = None,
) -> list[FillRenderData]:
    """Returns the document's fill areas with text templates resolved."""
    context = _text_context(document, text_values)
    entries = resolved_text_cache if resolved_text_cache is not None else {}
    resolved, _volatile = _resolve_text(document, context, entries)
    return _build_fills(document, resolved, set(exclude_ids or ()))


def _text_boxes(document: Any) -> list[TextBox]:
    return [
        entity
        for entity in document.registry.entities()
        if isinstance(entity, TextBox)
    ]


def _text_context(
    document: Any, text_values: dict[str, Any] | None
) -> dict[str, Any]:
    context = dict(document.solved_ctx or {})
    context.update(text_values or {})
    return context


def _read_names(
    document: Any, initial_values: dict[str, Any] | None
) -> set[str]:
    expressions = document.params.to_dict().get("expressions", {})
    names = set(initial_values or {}) | set(expressions)
    for expression in expressions.values():
        names |= names_in_expression(str(expression))
    for constraint in document.constraints:
        expression = getattr(constraint, "expression", None)
        if expression:
            names |= names_in_expression(str(expression))
    names |= variables_used(document)
    return names


def _effective_values(
    document: Any,
    params: dict[str, Any] | None,
    initial_values: dict[str, Any] | None,
    text_values: dict[str, Any] | None,
) -> dict[str, Any]:
    """The effective values without cloning the whole sketch."""
    context = ParameterContext.from_dict(document.params.to_dict())
    seed = dict(initial_values or {})
    seed.update(params or {})
    context.evaluate_all(seed)
    values = dict(context.get_all_values())
    values.update(text_values or {})
    return values


def _value_key(
    values: dict[str, Any],
    read_names: set[str],
    function_names: set[str],
) -> tuple:
    return tuple(
        sorted(
            (name, repr(values[name]))
            for name in read_names
            if name in values and name not in function_names
        )
    )


def _text_key(entries: dict) -> tuple:
    return tuple(
        sorted(
            (str(entity_id), repr(tuple(entry)))
            for entity_id, entry in entries.items()
        )
    )


def _resolve_text(
    document: Any,
    context: dict[str, Any],
    cache: dict,
) -> tuple[dict[int, str | None], bool]:
    """Resolves every text box, reusing current cache entries.

    Raygeo owns the placeholder-level stamps and reuse rules; this
    function only walks the document's text boxes and stores the
    returned entries.
    """
    functions = get_template_functions()
    resolved: dict[int, str | None] = {}
    volatile = False
    for entity in _text_boxes(document):
        value, entry, part_volatile = resolve_template(
            entity.content,
            context,
            functions,
            cache.get(entity.id),
        )
        cache[entity.id] = entry
        resolved[entity.id] = value
        volatile = volatile or part_volatile
    return resolved, volatile


def _build_fills(
    document: Any,
    resolved: dict[int, str | None],
    exclude_ids: set[int] | None = None,
) -> list[FillRenderData]:
    exclude_ids = exclude_ids or set()
    result: list[FillRenderData] = []
    for fill in document.fills:
        if not fill.boundary:
            continue
        geometry = document.create_fill_geometry(fill, exclude_ids)
        if geometry is not None:
            result.append(
                FillRenderData(
                    geometry=geometry,
                    style=fill.style,
                    color=fill.color,
                    gradient_stops=fill.gradient_stops,
                    gradient_angle=fill.gradient_angle,
                )
            )
    for entity in _text_boxes(document):
        if entity.id in exclude_ids or entity.construction:
            continue
        geometry = entity.to_geometry(
            document.registry, resolved_content=resolved.get(entity.id)
        )
        if geometry is None:
            continue
        color = (
            entity.fill_color
            if entity.fill_color is not None
            else DEFAULT_FILL_COLOR
        )
        result.append(
            FillRenderData(
                geometry=geometry,
                style=FillStyle.SOLID,
                color=color,
            )
        )
    return result
