"""Tests for assembly_warnings.translate_assembly_warning."""

from unittest.mock import patch

import pytest

from rayforge.pipeline import assembly_warnings
from rayforge.pipeline.assembly_warnings import translate_assembly_warning


@pytest.fixture(autouse=True)
def _identity_gettext(monkeypatch):
    """Force ``_()`` to the identity so templates are asserted verbatim."""
    monkeypatch.setattr(assembly_warnings, "_", lambda s: s)


class _FakeWarning:
    def __init__(self, kind, face_id="", region=None, detail=""):
        self.kind = kind
        self.face_id = face_id
        self.region = region
        self.detail = detail


def test_face_failed_with_named_face():
    from raygeo.ops.assembly import AssemblyWarningKind

    w = _FakeWarning(
        kind=AssemblyWarningKind.FACE_FAILED,
        face_id="1",
        detail="boom",
    )
    assert translate_assembly_warning(w) == (
        "Face '1' could not be machined: boom"
    )


def test_face_failed_default_face_uses_translated_label():
    from raygeo.ops.assembly import AssemblyWarningKind

    w = _FakeWarning(
        kind=AssemblyWarningKind.FACE_FAILED,
        face_id="",
        detail="boom",
    )
    assert translate_assembly_warning(w) == (
        "Face 'default face' could not be machined: boom"
    )


def test_region_failed_with_region_index():
    from raygeo.ops.assembly import AssemblyWarningKind

    w = _FakeWarning(
        kind=AssemblyWarningKind.REGION_FAILED,
        face_id="2",
        region=3,
        detail="stalled",
    )
    assert translate_assembly_warning(w) == (
        "Region 3 of face '2' could not be machined: stalled"
    )


def test_region_failed_without_region_index():
    from raygeo.ops.assembly import AssemblyWarningKind

    w = _FakeWarning(
        kind=AssemblyWarningKind.REGION_FAILED,
        face_id="",
        region=None,
        detail="stalled",
    )
    assert translate_assembly_warning(w) == (
        "Region ? of face 'default face' could not be machined: stalled"
    )


def test_unknown_kind_uses_generic_template():
    w = _FakeWarning(kind="SOME_OTHER_KIND", detail="weird")
    assert translate_assembly_warning(w) == ("Machining warning: weird")


def test_strings_are_marked_for_translation():
    """All templates pass through the module's ``_()``."""
    from raygeo.ops.assembly import AssemblyWarningKind

    seen = []

    def _capture(text):
        seen.append(text)
        return text

    with patch.object(assembly_warnings, "_", _capture):
        translate_assembly_warning(
            _FakeWarning(
                kind=AssemblyWarningKind.FACE_FAILED,
                face_id="1",
                detail="boom",
            )
        )
        translate_assembly_warning(
            _FakeWarning(
                kind=AssemblyWarningKind.REGION_FAILED,
                face_id="1",
                region=0,
                detail="stalled",
            )
        )
        translate_assembly_warning(_FakeWarning(kind="OTHER", detail="weird"))

    assert "Face '{face}' could not be machined: {detail}" in seen
    assert (
        "Region {region} of face '{face}' could not be machined: {detail}"
    ) in seen
    assert "Machining warning: {detail}" in seen
    assert "default face" in seen
