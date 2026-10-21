from unittest.mock import MagicMock

import pytest
from gi.repository import Gdk

from rayforge.ui_gtk.canvas.canvas import Canvas
from rayforge.ui_gtk.shared.keyboard import PRIMARY_MODIFIER_MASK


@pytest.fixture
def canvas():
    c = Canvas.__new__(Canvas)
    c.edit_context = MagicMock()
    c.edit_context.handle_edit_key.return_value = True
    return c


@pytest.mark.ui
def test_plain_keys_go_to_the_edit_context(canvas):
    assert canvas.on_key_pressed(None, Gdk.KEY_c, 0, 0) is True
    canvas.edit_context.handle_edit_key.assert_called_once_with(Gdk.KEY_c)


@pytest.mark.ui
def test_shortcuts_with_the_primary_modifier_pass_through(canvas):
    handled = canvas.on_key_pressed(None, Gdk.KEY_c, 0, PRIMARY_MODIFIER_MASK)
    assert handled is False
    canvas.edit_context.handle_edit_key.assert_not_called()


@pytest.mark.ui
def test_keys_the_edit_context_ignores_propagate(canvas):
    canvas.edit_context.handle_edit_key.return_value = False
    assert canvas.on_key_pressed(None, Gdk.KEY_z, 0, 0) is False
