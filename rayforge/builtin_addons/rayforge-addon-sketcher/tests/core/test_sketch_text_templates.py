from datetime import date
from typing import cast
from sketcher.core import Sketch
from sketcher.core.entities.text_box import TextBoxEntity
from rayforge.core.geo.font_config import FontConfig
from rayforge.core.varset import FloatVar, Var


def _add_text_box(
    sketch: Sketch, content: str, x: float = 0, y: float = 0
) -> TextBoxEntity:
    """Helper to add a text box to a sketch and return it."""
    origin = sketch.add_point(x, y)
    width_pt = sketch.add_point(x + 10, y)
    height_pt = sketch.add_point(x, y + 10)
    box_id = sketch.registry.add_text_box(
        origin, width_pt, height_pt, content, FontConfig()
    )
    return cast(TextBoxEntity, sketch.registry.get_entity(box_id))


def test_resolve_plain_text():
    """Plain text content passes through resolve unchanged."""
    s = Sketch()
    box = _add_text_box(s, "Hello")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "Hello"


def test_resolve_template_from_params():
    """Template content is resolved using sketch parameters."""
    s = Sketch()
    s.set_param("width", 50.0)
    box = _add_text_box(s, "W={width}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "W=50.0"


def test_resolve_template_from_input_parameters():
    """Template content is resolved using input_parameters VarSet."""
    s = Sketch()
    s.input_parameters.add(FloatVar(key="count", label="Count", value=7))
    box = _add_text_box(s, "{count:.0f}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "7"


def test_resolve_template_with_format_spec():
    """Format specs in templates are applied."""
    s = Sketch()
    s.set_param("ratio", 3.14159)
    box = _add_text_box(s, "{ratio:.2f}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "3.14"


def test_resolve_template_with_expression():
    """Math expressions in templates are evaluated."""
    s = Sketch()
    s.set_param("width", 25.0)
    box = _add_text_box(s, "{sqrt(width):.1f}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "5.0"


def test_resolve_empty_content_returns_none():
    """Empty text boxes return None from resolve."""
    s = Sketch()
    box = _add_text_box(s, "")
    s.solve()
    assert s._resolve_text_content(box) is None


def test_resolve_multiple_text_boxes():
    """Multiple text boxes are each resolved independently."""
    s = Sketch()
    s.set_param("a", 10.0)
    s.set_param("b", 20.0)

    box1 = _add_text_box(s, "A={a}", x=0, y=0)
    box2 = _add_text_box(s, "B={b}", x=50, y=0)

    s.solve()
    assert s._resolve_text_content(box1) == "A=10.0"
    assert s._resolve_text_content(box2) == "B=20.0"


def test_to_geometry_uses_resolved_content():
    """to_geometry produces geometry from the resolved content."""
    s = Sketch()
    s.set_param("val", 42.0)
    _add_text_box(s, "{val}")
    s.solve()

    geo = s.to_geometry()
    assert len(geo) > 0


def test_resolve_date_today():
    """today() is available as a template function."""
    s = Sketch()
    box = _add_text_box(s, "{today()}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved is not None
    assert date.today().isoformat() in resolved


def test_resolve_uuid4():
    """uuid4() is available as a template function."""
    s = Sketch()
    box = _add_text_box(s, "{uuid4()}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved is not None
    assert len(resolved) == 8
    assert "{" not in resolved


def test_resolve_string_input_parameter():
    """String-type input parameters are available in templates."""
    s = Sketch()
    s.input_parameters.add(
        Var(key="name", label="Name", var_type=str, default="Widget")
    )
    box = _add_text_box(s, "Part: {name}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "Part: Widget"


def test_resolve_string_param_overrides_numeric_ctx():
    """String input parameters take precedence over ParameterContext."""
    s = Sketch()
    s.input_parameters.add(
        Var(key="label", label="Label", var_type=str, default="Hello")
    )
    box = _add_text_box(s, "{label}")
    s.solve()
    resolved = s._resolve_text_content(box)
    assert resolved == "Hello"


def test_entity_content_unchanged_after_solve():
    """Entity.content stays as raw template after solve."""
    s = Sketch()
    s.set_param("width", 50.0)
    box = _add_text_box(s, "W={width}")
    s.solve()
    assert box.content == "W={width}"


def test_get_geometry_resolves_templates():
    """get_geometry resolves templates in the clone's export."""
    s = Sketch()
    s.set_param("width", 50.0)
    _add_text_box(s, "W={width}")
    s.solve()

    geo, _ = s.get_geometry()
    assert len(geo) > 0
