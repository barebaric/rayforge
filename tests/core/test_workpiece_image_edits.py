from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import pyvips

from rayforge.core.doc import Doc
from rayforge.core.image_edit import (
    ImageEditOp,
    ImageEditStep,
    image_edit_registry,
)
from rayforge.core.vectorization_spec import LayerImportMode, PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image import import_file

SVG = b"""<svg width="10mm" height="10mm" viewBox="0 0 10 10"
     xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="10" height="10" fill="#202020"/>
</svg>"""


class _BlankOp(ImageEditOp):
    NAME = "test_blank"
    LABEL = "Blank"
    calls = 0

    @classmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        cls.calls += 1
        return (image * 0 + 255).cast("uchar")


@pytest.fixture
def blank_op():
    image_edit_registry.register(_BlankOp, addon_name="test_wp_edits")
    _BlankOp.calls = 0
    yield _BlankOp
    image_edit_registry.unregister_all_from_addon("test_wp_edits")


@pytest.fixture
def workpiece(tmp_path: Path) -> WorkPiece:
    path = tmp_path / "square.svg"
    path.write_bytes(SVG)
    payload = import_file(
        path,
        vectorization_spec=PassthroughSpec(
            layer_import_mode=LayerImportMode.FLATTEN
        ),
    )
    assert payload is not None
    wp = cast(WorkPiece, payload.items[0])
    doc = Doc()
    doc.add_asset(payload.source)
    doc.active_layer.add_child(wp)
    return wp


def _mean_rgb(image: pyvips.Image) -> float:
    return image.extract_band(0, n=3).avg()


def test_new_workpiece_has_no_edits(workpiece):
    assert workpiece.image_edits == ()


def test_processed_image_without_edits_is_the_base_render(workpiece):
    base = workpiece._get_vips_image(20, 20)
    assert workpiece.get_processed_image(20, 20) is base


def test_processed_image_applies_the_stack(workpiece, blank_op):
    base_mean = _mean_rgb(workpiece._get_vips_image(20, 20))
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    assert _mean_rgb(workpiece.get_processed_image(20, 20)) == 255
    assert _mean_rgb(workpiece._get_vips_image(20, 20)) == base_mean


def test_render_to_pixels_uses_the_processed_image(workpiece, blank_op):
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    surface = workpiece.render_to_pixels(20, 20)
    assert surface is not None
    assert set(bytes(surface.get_data())) == {255}


def test_processed_image_is_cached(workpiece, blank_op):
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    first = workpiece.get_processed_image(20, 20)
    assert workpiece.get_processed_image(20, 20) is first
    assert blank_op.calls == 1


def test_setting_edits_invalidates_and_notifies(workpiece, blank_op):
    workpiece.get_processed_image(20, 20)
    seen = []
    workpiece.updated.connect(seen.append, weak=False)
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    assert seen == [workpiece]
    assert _mean_rgb(workpiece.get_processed_image(20, 20)) == 255


def test_setting_equal_edits_does_not_notify(workpiece, blank_op):
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    seen = []
    workpiece.updated.connect(seen.append, weak=False)
    workpiece.image_edits = (ImageEditStep(blank_op.NAME),)
    assert seen == []


def test_unregistered_op_is_skipped(workpiece):
    workpiece.image_edits = [ImageEditStep("not_installed")]
    base = workpiece._get_vips_image(20, 20)
    assert _mean_rgb(workpiece.get_processed_image(20, 20)) == _mean_rgb(base)


def test_edits_survive_serialization(workpiece):
    edits = (
        ImageEditStep("not_installed", {"x": 1}),
        ImageEditStep("test_blank", enabled=False),
    )
    workpiece.image_edits = edits
    restored = WorkPiece.from_dict(workpiece.to_dict())
    assert restored.image_edits == edits


def test_old_documents_load_without_edits(workpiece):
    data = workpiece.to_dict()
    data.pop("image_edits", None)
    assert WorkPiece.from_dict(data).image_edits == ()


def test_in_world_copies_the_edits(workpiece):
    workpiece.image_edits = [ImageEditStep("test_blank")]
    assert workpiece.in_world().image_edits == workpiece.image_edits
