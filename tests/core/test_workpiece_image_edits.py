from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

import pytest
import pyvips

from rayforge.core.doc import Doc
from rayforge.core.image_cache import (
    IMAGE_CACHE_MAX_BYTES,
    IMAGE_CACHE_MAX_ENTRIES,
    image_nbytes,
)
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


class _ResolutionOp(ImageEditOp):
    NAME = "test_resolution"
    LABEL = "Resolution"
    seen: ClassVar[list[tuple[float, float]]] = []

    @classmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        cls.seen.append((image.xres, image.yres))
        return image


@pytest.fixture
def resolution_op():
    image_edit_registry.register(_ResolutionOp, addon_name="test_wp_res")
    _ResolutionOp.seen = []
    yield _ResolutionOp
    image_edit_registry.unregister_all_from_addon("test_wp_res")


def test_edits_see_the_resolution_in_pixels_per_mm(workpiece, resolution_op):
    workpiece.set_size(10.0, 5.0)
    workpiece.image_edits = [ImageEditStep(resolution_op.NAME)]
    workpiece.get_processed_image(40, 10)
    assert resolution_op.seen == [pytest.approx((4.0, 2.0))]


def test_resizing_reapplies_the_edits(workpiece, resolution_op):
    workpiece.image_edits = [ImageEditStep(resolution_op.NAME)]
    workpiece.set_size(10.0, 10.0)
    workpiece.get_processed_image(20, 20)
    workpiece.set_size(20.0, 20.0)
    workpiece.get_processed_image(20, 20)
    assert resolution_op.seen == [
        pytest.approx((2.0, 2.0)),
        pytest.approx((1.0, 1.0)),
    ]


def test_slider_changes_keep_one_processed_image(workpiece, blank_op):
    for value in range(20):
        workpiece.image_edits = [ImageEditStep(blank_op.NAME, {"v": value})]
        workpiece.get_processed_image(20, 20)
    assert len(workpiece._processed_cache) == 1


def test_zooming_keeps_a_bounded_number_of_images(workpiece, blank_op):
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    for zoom in range(1, 11):
        workpiece.get_processed_image(10 * zoom, 10 * zoom)
    assert len(workpiece._render_cache) == IMAGE_CACHE_MAX_ENTRIES
    assert len(workpiece._processed_cache) == IMAGE_CACHE_MAX_ENTRIES
    assert (100, 100) in workpiece._render_cache
    assert (10, 10) not in workpiece._render_cache


def test_resizing_keeps_a_bounded_number_of_images(workpiece, blank_op):
    workpiece.image_edits = [ImageEditStep(blank_op.NAME)]
    for step in range(1, 11):
        workpiece.set_size(10.0 * step, 10.0)
        workpiece.get_processed_image(20, 20)
    assert len(workpiece._processed_cache) <= IMAGE_CACHE_MAX_ENTRIES


def test_image_caches_have_a_byte_budget(workpiece):
    for cache in (workpiece._render_cache, workpiece._processed_cache):
        assert cache.max_bytes == IMAGE_CACHE_MAX_BYTES


def test_byte_budget_evicts_the_least_recently_used(workpiece):
    workpiece._render_cache.max_bytes = 3600
    workpiece._get_vips_image(20, 20)
    workpiece._get_vips_image(10, 10)
    workpiece._get_vips_image(20, 20)
    workpiece._get_vips_image(22, 22)
    assert set(workpiece._render_cache.keys()) == {(20, 20), (22, 22)}
    assert workpiece._render_cache.nbytes <= 3600


def test_cached_size_is_the_uncompressed_pixel_size():
    image = pyvips.Image.black(30, 20, bands=4).cast("ushort")
    assert image_nbytes(image) == 30 * 20 * 4 * 2


def test_source_base_images_have_a_byte_budget(workpiece):
    source = workpiece.source
    assert source is not None
    assert source._base_image_cache.max_bytes == IMAGE_CACHE_MAX_BYTES
    big = pyvips.Image.black(4000, 3000, bands=4)
    for step in range(10):
        source.cache_base_image(step, 4000, 3000, big)
    assert source._base_image_cache.nbytes <= IMAGE_CACHE_MAX_BYTES
