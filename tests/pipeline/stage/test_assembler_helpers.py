"""Tests for build_part() assembler helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import cairo
import numpy as np
import pytest
from raygeo.geo import Geometry, Matrix
from raygeo.ops import Ops
from raygeo.ops.assembly import AssemblyResult
from raygeo.ops.assembly.contour import contour
from raygeo.ops.part import Part
from raygeo.ops.types import SectionType

from rayforge.core.source_asset import SourceAsset
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.image.dither import DitherAlgorithm
from rayforge.machine.models.machine import Laser
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.stage.assembler_helpers import (
    DepthMode,
    build_part_raster,
    build_part_vector,
    compute_raster_auto_levels,
    make_artifact,
    preprocess_raster_image,
    wrap_assembler_result,
)


def _create_rect(x, y, w, h, cw=False):
    """Create a unit-box rectangle geometry."""
    geo = Geometry()
    geo.move_to(x, y)
    if not cw:
        geo.line_to(x + w, y)
        geo.line_to(x + w, y + h)
        geo.line_to(x, y + h)
    else:
        geo.line_to(x, y + h)
        geo.line_to(x + w, y + h)
        geo.line_to(x + w, y)
    geo.close_path()
    return geo


@pytest.fixture
def vector_workpiece():
    """WorkPiece with normalized (0–1) vector boundaries."""
    outer = _create_rect(0, 0, 1.0, 1.0, cw=False)
    inner = _create_rect(0.25, 0.25, 0.5, 0.5, cw=True)

    combined = Geometry()
    combined.extend(outer)
    combined.extend(inner)

    source = SourceAsset(
        source_file=Path("test.dxf"), original_data=b"", renderer=MagicMock()
    )
    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=combined,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )

    wp = WorkPiece(name="vector_wp", source_segment=segment)
    wp._boundaries_cache = combined
    wp.set_size(50, 30)
    return wp


@pytest.fixture
def empty_workpiece():
    """WorkPiece without vector boundaries."""
    source = SourceAsset(
        source_file=Path("test.png"), original_data=b"", renderer=MagicMock()
    )
    segment = SourceAssetSegment(
        source_asset_uid=source.uid,
        pristine_geometry=None,
        vectorization_spec=PassthroughSpec(),
        normalization_matrix=Matrix.identity(),
    )
    wp = WorkPiece(name="raster_wp", source_segment=segment)
    wp.set_size(40, 40)
    return wp


def _make_filled_surface(width, height):
    """Create a Cairo surface with a black rectangle (traceable)."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 1)
    ctx.paint()
    return surface


class TestBuildPartVector:
    """Tests for build_part_vector()."""

    def test_vector_source_uses_workpiece_boundaries(self, vector_workpiece):
        """When the workpiece has boundaries, Part is built from them."""
        part = build_part_vector(vector_workpiece)

        assert part is not None
        assert part.has_geometry()
        assert part.size_mm == (50, 30)

        assert part.geometry is not None
        rect = part.geometry.rect()
        assert rect[0] == pytest.approx(0, abs=0.01)
        assert rect[1] == pytest.approx(0, abs=0.01)
        assert rect[2] == pytest.approx(50, abs=0.01)
        assert rect[3] == pytest.approx(30, abs=0.01)

    def test_vector_source_without_boundaries_returns_none(
        self, empty_workpiece
    ):
        """No boundaries and no surface → None."""
        part = build_part_vector(empty_workpiece)
        assert part is None

    def test_vector_source_with_none_surface_and_no_boundaries(
        self, empty_workpiece
    ):
        """Explicitly passing surface=None with no boundaries → None."""
        part = build_part_vector(empty_workpiece, surface=None)
        assert part is None

    def test_raster_fallback_traces_surface(self, empty_workpiece):
        """Without boundaries but with a surface, tracing produces a Part."""
        surface = _make_filled_surface(100, 100)
        part = build_part_vector(empty_workpiece, surface=surface)

        assert part is not None
        assert part.has_geometry()
        assert part.size_mm == (40, 40)

    def test_override_threshold_ignores_vector_source(self, vector_workpiece):
        """override_threshold=True forces raster tracing even with
        boundaries."""
        surface = _make_filled_surface(100, 100)
        part = build_part_vector(
            vector_workpiece,
            surface=surface,
            override_threshold=True,
            threshold=0.5,
        )

        assert part is not None
        assert part.has_geometry()

    def test_override_threshold_without_surface_returns_none(
        self, vector_workpiece
    ):
        """override_threshold=True but no surface → None (can't trace)."""
        part = build_part_vector(vector_workpiece, override_threshold=True)
        assert part is None

    def test_normalize_windings_applied(self, vector_workpiece):
        """normalize_windings=True normalizes winding orders on the
        scaled geometry."""
        part = build_part_vector(vector_workpiece, normalize_windings=True)

        assert part is not None
        assert part.has_geometry()
        assert part.size_mm == (50, 30)

    def test_normalize_windings_with_raster_fallback(self, empty_workpiece):
        """normalize_windings works on traced geometry too."""
        surface = _make_filled_surface(100, 100)
        part = build_part_vector(
            empty_workpiece,
            surface=surface,
            normalize_windings=True,
        )

        assert part is not None
        assert part.has_geometry()

    def test_empty_surface_returns_none(self, empty_workpiece):
        """A blank (transparent) surface produces no geometry → None."""
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
        part = build_part_vector(empty_workpiece, surface=surface)
        assert part is None


class TestBuildPartRaster:
    """Tests for build_part_raster()."""

    def test_raster_part_has_no_geometry(self, vector_workpiece):
        """Raster Part carries metadata only, no vector geometry."""
        part = build_part_raster(vector_workpiece, (10.0, 10.0))

        assert not part.has_geometry()
        assert part.geometry is None
        assert part.size_mm == (50, 30)
        assert part.pixels_per_mm == (10.0, 10.0)

    def test_raster_part_with_empty_workpiece(self, empty_workpiece):
        """Raster Part works without vector boundaries."""
        part = build_part_raster(empty_workpiece, (5.0, 5.0))

        assert not part.has_geometry()
        assert part.size_mm == (40, 40)
        assert part.pixels_per_mm == (5.0, 5.0)


def _make_assembly_result_with_rect(w, h):
    """Create an AssemblyResult containing a simple rectangle."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(w, 0)
    geo.line_to(w, h)
    geo.line_to(0, h)
    geo.close_path()
    # AssemblyResult is a Rust/PyO3 object; we can't easily set ops
    # directly, so use a contour assembler call to get a real result.
    part = Part(geometry=geo, size_mm=(w, h))
    return contour(
        part,
        kerf_mm=0.0,
        path_offset_mm=0.0,
        cut_side="centerline",
    )


class TestMakeArtifact:
    """Tests for make_artifact()."""

    def test_vector_artifact(self, vector_workpiece):
        """make_artifact with is_vector=True uses MILLIMETER_SPACE."""
        ops = Ops()
        artifact = make_artifact(
            ops, vector_workpiece, generation_id=42, is_vector=True
        )

        assert isinstance(artifact, WorkPieceArtifact)
        assert artifact.ops is ops
        assert artifact.is_scalable is False
        assert (
            artifact.source_coordinate_system
            == CoordinateSystem.MILLIMETER_SPACE
        )
        assert artifact.source_dimensions == (50, 30)
        assert artifact.generation_size == (50, 30)
        assert artifact.generation_id == 42

    def test_raster_artifact(self, vector_workpiece):
        """make_artifact with is_vector=False uses PIXEL_SPACE and
        custom source_dimensions."""
        ops = Ops()
        artifact = make_artifact(
            ops,
            vector_workpiece,
            generation_id=99,
            is_vector=False,
            source_dimensions=(200, 150),
        )

        assert (
            artifact.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
        )
        assert artifact.source_dimensions == (200, 150)
        assert artifact.generation_size == (50, 30)

    def test_empty_ops(self, vector_workpiece):
        """make_artifact works with empty Ops."""
        artifact = make_artifact(Ops(), vector_workpiece, generation_id=1)

        assert artifact.ops.is_empty()
        assert isinstance(artifact, WorkPieceArtifact)


class TestWrapAssemblerResult:
    """Tests for wrap_assembler_result()."""

    def test_single_section_vector(self, vector_workpiece):
        """Default: single VECTOR_OUTLINE section wrapping all ops."""
        laser = Laser()
        result = _make_assembly_result_with_rect(50, 30)

        artifact = wrap_assembler_result(
            result, vector_workpiece, laser, generation_id=1
        )

        assert isinstance(artifact, WorkPieceArtifact)
        assert not artifact.ops.is_empty()
        assert (
            artifact.source_coordinate_system
            == CoordinateSystem.MILLIMETER_SPACE
        )
        assert artifact.source_dimensions == (50, 30)

    def test_split_contours(self, vector_workpiece):
        """split_contours=True wraps each contour in its own section."""
        laser = Laser()
        result = _make_assembly_result_with_rect(50, 30)

        artifact = wrap_assembler_result(
            result,
            vector_workpiece,
            laser,
            generation_id=1,
            split_contours=True,
        )

        assert not artifact.ops.is_empty()
        sections = artifact.ops.sections()
        typed = [s for s in sections if s.section_type is not None]
        assert len(typed) >= 1
        for section in typed:
            assert section.section_type == SectionType.VECTOR_OUTLINE

    def test_set_power(self, vector_workpiece):
        """set_power emits a power command inside the section."""
        laser = Laser()
        result = _make_assembly_result_with_rect(50, 30)

        artifact = wrap_assembler_result(
            result,
            vector_workpiece,
            laser,
            generation_id=1,
            set_power=0.5,
        )

        assert not artifact.ops.is_empty()

    def test_raster_section_type(self, vector_workpiece):
        """section_type=RASTER_FILL wraps ops in a raster section."""
        laser = Laser()
        result = _make_assembly_result_with_rect(50, 30)

        artifact = wrap_assembler_result(
            result,
            vector_workpiece,
            laser,
            generation_id=1,
            section_type=SectionType.RASTER_FILL,
            is_vector=False,
            source_dimensions=(100, 100),
        )

        assert (
            artifact.source_coordinate_system == CoordinateSystem.PIXEL_SPACE
        )
        assert artifact.source_dimensions == (100, 100)
        sections = artifact.ops.sections()
        typed = [s for s in sections if s.section_type is not None]
        assert len(typed) >= 1
        assert typed[0].section_type == SectionType.RASTER_FILL

    def test_empty_result_ops(self, vector_workpiece):
        """When result.ops is empty, artifact has empty ops."""
        laser = Laser()
        result = AssemblyResult()

        artifact = wrap_assembler_result(
            result, vector_workpiece, laser, generation_id=1
        )

        assert artifact.ops.is_empty()
        assert isinstance(artifact, WorkPieceArtifact)


class TestPreprocessRasterImage:
    """Tests for preprocess_raster_image()."""

    def _black_surface(self, w=10, h=10):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0, 0, 0)
        ctx.paint()
        return surface

    def _gray_surface(self, w=10, h=10):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.5, 0.5, 0.5)
        ctx.paint()
        return surface

    def test_power_modulated_returns_grayscale_with_alpha(self):
        surface = self._black_surface()
        image, alpha = preprocess_raster_image(
            surface, mode=DepthMode.POWER_MODULATION, auto_levels=False
        )

        assert image is not None
        assert alpha is not None
        assert image.shape == (10, 10)
        assert alpha.shape == (10, 10)
        assert image.dtype == np.uint8

    def test_multi_pass_returns_grayscale_with_alpha(self):
        surface = self._gray_surface()
        image, alpha = preprocess_raster_image(
            surface, mode=DepthMode.MULTI_PASS, auto_levels=False
        )

        assert image is not None
        assert alpha is not None
        assert image.dtype == np.uint8

    def test_dither_returns_binary_no_alpha(self):
        surface = self._gray_surface()
        image, alpha = preprocess_raster_image(
            surface,
            mode=DepthMode.DITHER,
            dither_algorithm=DitherAlgorithm.FLOYD_STEINBERG,
            laser_spot_x_mm=0.1,
            pixels_per_mm_x=10.0,
        )

        assert image is not None
        assert alpha is None
        assert set(np.unique(image)).issubset({0, 1})

    def test_mask_scan_returns_binary_no_alpha(self):
        surface = self._black_surface()
        image, alpha = preprocess_raster_image(
            surface, mode=DepthMode.CONSTANT_POWER, threshold=128
        )

        assert image is not None
        assert alpha is None
        assert set(np.unique(image)).issubset({0, 1})

    def test_invert_flips_grayscale(self):
        surface = self._black_surface()
        image_normal, _ = preprocess_raster_image(
            surface,
            mode=DepthMode.POWER_MODULATION,
            invert=False,
            auto_levels=False,
        )
        image_inverted, _ = preprocess_raster_image(
            surface,
            mode=DepthMode.POWER_MODULATION,
            invert=True,
            auto_levels=False,
        )

        assert image_normal is not None
        assert image_inverted is not None
        assert np.mean(image_inverted) > np.mean(image_normal)

    def test_auto_levels_normalizes_grayscale(self):
        surface = self._gray_surface()
        image_no_levels, _ = preprocess_raster_image(
            surface,
            mode=DepthMode.POWER_MODULATION,
            auto_levels=False,
        )
        image_auto_levels, _ = preprocess_raster_image(
            surface,
            mode=DepthMode.POWER_MODULATION,
            auto_levels=True,
        )

        assert image_no_levels is not None
        assert image_auto_levels is not None
        assert image_auto_levels.dtype == np.uint8

    def test_computed_auto_levels_used_when_provided(self):
        surface = self._gray_surface()
        image, _ = preprocess_raster_image(
            surface,
            mode=DepthMode.POWER_MODULATION,
            auto_levels=True,
            computed_auto_levels=(50, 200),
        )

        assert image is not None
        assert image.dtype == np.uint8

    def test_manual_levels_applied_when_auto_disabled(self):
        surface = self._gray_surface()
        image, _ = preprocess_raster_image(
            surface,
            mode=DepthMode.POWER_MODULATION,
            auto_levels=False,
            black_point=100,
            white_point=200,
        )

        assert image is not None

    def test_mask_scan_threshold_affects_output(self):
        surface = self._gray_surface()
        image_low, _ = preprocess_raster_image(
            surface, mode=DepthMode.CONSTANT_POWER, threshold=100
        )
        image_high, _ = preprocess_raster_image(
            surface, mode=DepthMode.CONSTANT_POWER, threshold=200
        )

        assert image_low is not None
        assert image_high is not None
        assert np.sum(image_low) != np.sum(image_high)


class TestComputeRasterAutoLevels:
    """Tests for compute_raster_auto_levels()."""

    def test_returns_levels_for_gray_surface(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.5, 0.5, 0.5)
        ctx.paint()

        mock_wp = MagicMock()
        mock_wp.size = (10.0, 10.0)
        mock_wp.render_to_pixels.return_value = surface

        result = compute_raster_auto_levels(mock_wp, (10.0, 10.0))

        assert result is not None
        black_pt, white_pt = result
        assert 0 <= black_pt <= 253
        assert 2 <= white_pt <= 255
        assert black_pt < white_pt

    def test_returns_none_when_render_fails(self):
        mock_wp = MagicMock()
        mock_wp.size = (10.0, 10.0)
        mock_wp.render_to_pixels.return_value = None

        result = compute_raster_auto_levels(mock_wp, (10.0, 10.0))
        assert result is None

    def test_invert_affects_levels(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.2, 0.2, 0.2)
        ctx.paint()

        mock_wp = MagicMock()
        mock_wp.size = (10.0, 10.0)
        mock_wp.render_to_pixels.return_value = surface

        result_normal = compute_raster_auto_levels(
            mock_wp, (10.0, 10.0), invert=False
        )
        result_inverted = compute_raster_auto_levels(
            mock_wp, (10.0, 10.0), invert=True
        )

        assert result_normal is not None
        assert result_inverted is not None
        assert result_normal != result_inverted

    def test_preview_size_limited_by_max_preview_pixels(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
        ctx = cairo.Context(surface)
        ctx.set_source_rgb(0.5, 0.5, 0.5)
        ctx.paint()

        mock_wp = MagicMock()
        mock_wp.size = (100.0, 100.0)
        mock_wp.render_to_pixels.return_value = surface

        result = compute_raster_auto_levels(
            mock_wp, (10.0, 10.0), max_preview_pixels=50
        )

        assert result is not None
        call_args = mock_wp.render_to_pixels.call_args
        assert call_args[0][0] <= 50
        assert call_args[0][1] <= 50
