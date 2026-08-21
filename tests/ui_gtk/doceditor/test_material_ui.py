"""UI tests for material swatches and the add/edit material dialog."""

from pathlib import Path

import pytest
import pyvips
from gi.repository import Adw, Gtk

from rayforge.core.material import Material, MaterialAppearance
from rayforge.ui_gtk.doceditor.add_material_dialog import AddMaterialDialog
from rayforge.ui_gtk.shared.texture_loader import create_material_swatch

pytestmark = pytest.mark.ui


def _write_webp(path: Path, size: int = 64) -> None:
    """Write a small solid-color WebP image."""
    image = pyvips.Image.black(size, size).addalpha()
    image = image.colourspace("srgb")
    image.webpsave(str(path), Q=90)


def _textured_material(tmp_path: Path, texture_name: str) -> Material:
    _write_webp(tmp_path / texture_name)
    return Material(
        uid="tex_mat",
        appearance=MaterialAppearance(texture=texture_name),
        file_path=tmp_path / "tex_mat.yaml",
    )


class TestMaterialSwatch:
    """Test cases for the material swatch widget."""

    def test_swatch_shows_texture(self, tmp_path):
        """A material with a WebP texture shows a texture picture."""
        material = _textured_material(tmp_path, "wood.webp")

        widget = create_material_swatch(material)

        assert isinstance(widget, Gtk.Image)

    def test_swatch_missing_texture_file(self, tmp_path):
        """A declared texture that is missing falls back to color."""
        material = Material(
            uid="tex_mat",
            appearance=MaterialAppearance(texture="missing.webp"),
            file_path=tmp_path / "tex_mat.yaml",
        )

        widget = create_material_swatch(material)

        assert isinstance(widget, Gtk.Image)

    def test_swatch_falls_back_to_color(self, tmp_path):
        """A material without a texture shows a color swatch."""
        material = Material(
            uid="plain_mat",
            appearance=MaterialAppearance(color="#ff0000"),
            file_path=tmp_path / "plain_mat.yaml",
        )

        widget = create_material_swatch(material)

        assert isinstance(widget, Gtk.Image)

    def test_swatch_sizes_match(self, tmp_path):
        """
        Textured and color swatches allocate the same size.

        Gtk.Image paints the paintable at the pixel size, so both
        swatch kinds must end up at the same requested size.
        """
        textured = create_material_swatch(
            _textured_material(tmp_path, "wood.webp")
        )
        plain = create_material_swatch(
            Material(
                uid="plain_mat",
                appearance=MaterialAppearance(color="#ff0000"),
                file_path=tmp_path / "plain_mat.yaml",
            )
        )

        _, tex_nat = textured.get_preferred_size()
        _, plain_nat = plain.get_preferred_size()

        assert (tex_nat.width, tex_nat.height) == (32, 32)
        assert (plain_nat.width, plain_nat.height) == (32, 32)


class TestAddMaterialDialog:
    """Test cases for the add/edit material dialog."""

    def test_add_mode_defaults(self):
        """Add mode reports default PBR values and no texture."""
        dialog = AddMaterialDialog()

        data = dialog.get_material_data()

        assert data["roughness"] == 0.8
        assert data["metallic"] == 0.0
        assert data["texture_size_mm"] == 300.0
        assert data["texture"] is None

    def test_scale_row_disabled_without_texture(self):
        """The texture scale row is only usable when a texture is set."""
        dialog = AddMaterialDialog()

        assert dialog.texture_scale_row.get_sensitive() is False

    def test_pbr_rows_are_sliders(self):
        """Roughness and metallic are edited with sliders."""
        dialog = AddMaterialDialog()

        assert isinstance(dialog.roughness_row, Adw.ActionRow)
        assert isinstance(dialog.roughness_scale, Gtk.Scale)
        assert dialog.roughness_scale.get_value() == 0.8
        assert isinstance(dialog.metallic_row, Adw.ActionRow)
        assert isinstance(dialog.metallic_scale, Gtk.Scale)
        assert dialog.metallic_scale.get_value() == 0.0

    def test_edit_mode_prefills(self, tmp_path):
        """Edit mode pre-fills PBR values and texture from the material."""
        _write_webp(tmp_path / "wood.webp")
        material = Material(
            uid="tex_mat",
            name="Oak",
            appearance=MaterialAppearance(
                color="#A0522D",
                texture="wood.webp",
                texture_size_mm=250,
                roughness=0.55,
                metallic=0.05,
            ),
            file_path=tmp_path / "tex_mat.yaml",
        )

        dialog = AddMaterialDialog(material=material)

        data = dialog.get_material_data()

        assert data["name"] == "Oak"
        assert data["roughness"] == 0.55
        assert data["metallic"] == 0.05
        assert data["texture_size_mm"] == 250
        assert data["texture"] == tmp_path / "wood.webp"
        assert dialog.texture_scale_row.get_sensitive() is True

    def test_texture_scale_sensitivity_follows_texture(self):
        """The texture scale row is only active when a texture is set."""
        dialog = AddMaterialDialog()
        assert dialog.texture_scale_row.get_sensitive() is False

        dialog._texture_path = Path("/tmp/wood.webp")
        dialog._update_sensitivity()

        assert dialog.texture_scale_row.get_sensitive() is True

    def test_unset_color_reports_none(self):
        """Unsetting the color marks the material as not tinted."""
        dialog = AddMaterialDialog()
        dialog._color = "#FF0000"
        dialog._on_clear_color(Gtk.Button())

        assert dialog.get_color_hex() is None
        assert dialog.get_material_data()["color"] is None

    def test_edit_mode_prefills_unset_color(self, tmp_path):
        """Edit mode restores an unset ('not tinted') color."""
        _write_webp(tmp_path / "wood.webp")
        material = Material(
            uid="tex_mat",
            name="ABS",
            appearance=MaterialAppearance(
                color=None,
                texture="wood.webp",
                texture_size_mm=300,
            ),
            file_path=tmp_path / "tex_mat.yaml",
        )

        dialog = AddMaterialDialog(material=material)

        assert dialog.get_material_data()["color"] is None
