"""A small geometry provider used by job-counter tests."""

from __future__ import annotations

from typing import Any

from blinker import Signal
from raygeo.geo import Geometry
from raygeo.template import resolve_text as resolve_template

from rayforge.core.doc import Doc
from rayforge.core.expression import referenced_template_names
from rayforge.core.workpiece import WorkPiece


class TemplateProvider:
    """Renders a template to a rectangle whose width follows the text."""

    def __init__(self, template: str = ""):
        self.uid = "template-provider-uid"
        self.name = "template"
        self.template = template
        self.updated = Signal()
        self.calls: list[tuple[dict, dict]] = []

    @property
    def provider_type_name(self) -> str:
        return "template"

    @property
    def renderer(self):
        return None

    def get_geometry(
        self, params=None, *, text_values=None, resolved_text_cache=None
    ):
        self.calls.append((dict(params or {}), dict(text_values or {})))
        context = dict(params or {})
        context.update(text_values or {})
        resolved, entry, _volatile = resolve_template(
            self.template, context, {}, None
        )
        if resolved_text_cache is not None:
            resolved_text_cache[1] = entry
        width = max(float(len(resolved or "")), 1.0)
        height = 5.0
        geo = Geometry()
        geo.move_to(0.0, 0.0)
        geo.line_to(width, 0.0)
        geo.line_to(width, height)
        geo.line_to(0.0, height)
        geo.close_path()
        return geo, []

    def variables_used(self) -> set[str]:
        return referenced_template_names([self.template])

    def get_reference_point(self):
        return (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "template", "template": self.template}


def place(
    doc: Doc, template: str = "{counter}", offset: int = 0
) -> tuple[WorkPiece, TemplateProvider]:
    """Adds a workpiece whose provider renders *template*."""
    provider = TemplateProvider(template)
    wp = WorkPiece.from_geometry_provider(provider)
    wp._transient_geometry_provider = provider
    wp.variable_offset = offset
    doc.active_setup.add_child(wp)
    return wp, provider
