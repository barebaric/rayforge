"""A fake geometry provider that renders one text template."""

import uuid
from typing import Any, ClassVar

from blinker import Signal
from raygeo.geo import Geometry

from rayforge.core.doc import Doc
from rayforge.core.expression import ExpressionMap
from rayforge.core.workpiece import WorkPiece


class TemplateProvider:
    """A geometry provider rendering one template as a rectangle whose
    width grows with the length of the resolved text."""

    is_addable: ClassVar[bool] = False
    asset_type_name: ClassVar[str] = "template-test"
    display_icon_name: ClassVar[str] = ""
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = False
    type_display_name: ClassVar[str] = "Template"
    can_edit: ClassVar[bool] = False
    add_action: ClassVar[str | None] = None
    activate_action: ClassVar[str | None] = None
    edit_item_action: ClassVar[str | None] = None

    def __init__(self, template: str):
        self.template = template
        self._uid = str(uuid.uuid4())
        self.name = "template"
        self.updated = Signal()
        self.hidden = False
        self.calls: list[dict[str, Any]] = []
        self.rendered: list[str] = []

    @property
    def uid(self) -> str:
        return self._uid

    @property
    def provider_type_name(self) -> str:
        return "template-test"

    @property
    def renderer(self):
        return None

    def get_template_sources(self) -> list[str]:
        return [self.template]

    def get_geometry(self, params=None, *, resolved_text_cache=None):
        params = dict(params or {})
        self.calls.append(params)
        text = ExpressionMap(params).format(self.template)
        self.rendered.append(text)
        width = 2.0 * len(text)
        geo = Geometry()
        geo.move_to(0.0, 0.0)
        geo.line_to(width, 0.0)
        geo.line_to(width, 5.0)
        geo.line_to(0.0, 5.0)
        geo.close_path()
        return geo, []

    def to_dict(self) -> dict[str, Any]:
        return {"type": "template-test", "uid": self._uid}

    @classmethod
    def from_dict(cls, data):
        return cls("")

    def get_thumbnail(self, size: int):
        return None


def place(doc: Doc, template: str, offset: int = 0) -> WorkPiece:
    provider = TemplateProvider(template)
    doc.add_asset(provider)
    wp = WorkPiece.from_geometry_provider(provider)
    wp.variable_offset = offset
    doc.active_layer.add_child(wp)
    return wp
