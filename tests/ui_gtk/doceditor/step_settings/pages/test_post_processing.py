"""Tests for the PostProcessingPage."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from gi.repository import Gtk

from rayforge.ui_gtk.doceditor.step_settings.pages import PostProcessingPage


def _find_label(widget):
    """Iteratively search a widget tree for a Gtk.Label.

    Uses a visited set so a cyclic widget tree can never loop without
    bound. The placeholder's text is translated, so assert on type.
    """
    pending = [widget]
    visited = {hash(widget)}
    while pending:
        current = pending.pop()
        if isinstance(current, Gtk.Label):
            return True
        for child in current:
            if hash(child) not in visited:
                visited.add(hash(child))
                pending.append(child)
    return False


@pytest.mark.ui
def test_empty_step_shows_placeholder(editor, step):
    page = PostProcessingPage(editor, step)
    assert page._has_expanders is False
    assert _find_label(page._main_group)
