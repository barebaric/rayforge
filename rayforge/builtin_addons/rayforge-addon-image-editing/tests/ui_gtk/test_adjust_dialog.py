# flake8: noqa: E402
"""UI tests for the Adjust Image dialog."""

import gi
import pytest
import pyvips

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from image_editing.adjust_dialog import AdjustImageDialog
from image_editing.adjustments import ImageAdjustments
from image_editing.ops import INVERT

from rayforge.core.image_edit import ImageEditStep
from rayforge.core.undo import HistoryManager
from rayforge.core.workpiece import WorkPiece

TONE = ImageEditStep(
    "tone", {"brightness": 10.0, "contrast": 0.0, "gamma": 1.0}
)


@pytest.fixture
def workpieces():
    a, b = WorkPiece("a"), WorkPiece("b")
    a.image_edits = [TONE]
    b.image_edits = [INVERT]
    return [a, b]


@pytest.fixture
def dialog(ui_context, workpieces):
    dlg = AdjustImageDialog(None, HistoryManager(), workpieces)
    dlg.present()
    yield dlg
    dlg.destroy()


@pytest.mark.ui
def test_opens_with_the_first_workpiece_values(dialog):
    assert dialog.get_adjustments() == ImageAdjustments(brightness=10)


@pytest.mark.ui
def test_is_not_modal(dialog):
    assert not dialog.get_modal()


@pytest.mark.ui
def test_preview_shows_without_undo(dialog, workpieces):
    dialog.contrast_scale.set_value(20)
    dialog.preview()
    a, b = workpieces
    assert a.image_edits[0].params["contrast"] == 20
    assert b.image_edits[0] == INVERT
    assert b.image_edits[1].params["contrast"] == 20
    assert not dialog._history.can_undo()


@pytest.mark.ui
def test_closing_without_apply_restores_the_edits(dialog, workpieces):
    dialog.sharpen_scale.set_value(100)
    dialog.preview()
    dialog.close()
    assert workpieces[0].image_edits == (TONE,)
    assert workpieces[1].image_edits == (INVERT,)


@pytest.mark.ui
def test_apply_is_one_undo_step(dialog, workpieces):
    history = dialog._history
    dialog.gamma_scale.set_value(1.5)
    dialog.preview()
    dialog.apply()
    assert workpieces[0].image_edits[0].params["gamma"] == 1.5
    assert workpieces[1].image_edits[1].params["gamma"] == 1.5
    history.undo()
    assert workpieces[0].image_edits == (TONE,)
    assert workpieces[1].image_edits == (INVERT,)


@pytest.mark.ui
def test_reset_removes_the_adjustments(dialog, workpieces):
    dialog.reset()
    dialog.apply()
    assert workpieces[0].image_edits == ()
    assert workpieces[1].image_edits == (INVERT,)


def _cache_a_render(wp: WorkPiece) -> None:
    wp._processed_cache.put((10, 10, wp.size), pyvips.Image.black(10, 10))


@pytest.mark.ui
def test_previews_do_not_accumulate_cached_images(dialog, workpieces):
    for value in range(10):
        for wp in workpieces:
            _cache_a_render(wp)
        dialog.brightness_scale.set_value(value)
        dialog.preview()
    assert all(len(wp._processed_cache) == 0 for wp in workpieces)


@pytest.mark.ui
def test_closing_drops_the_preview_images(dialog, workpieces):
    dialog.sharpen_scale.set_value(100)
    dialog.preview()
    for wp in workpieces:
        _cache_a_render(wp)
    dialog.close()
    assert all(len(wp._processed_cache) == 0 for wp in workpieces)
