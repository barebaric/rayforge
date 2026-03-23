from unittest.mock import MagicMock, patch
import pytest

from sketcher.core.commands import GridCommand
from sketcher.ui_gtk.tools import GridTool


@pytest.fixture
def mock_element():
    element = MagicMock()
    element.sketch = MagicMock()
    element.editor = MagicMock()
    element.editor.parent_window = MagicMock()
    element.execute_command = MagicMock()
    element.selection.entity_ids = []
    return element


@pytest.fixture
def grid_tool(mock_element):
    return GridTool(mock_element)


@pytest.mark.ui
def test_grid_tool_initialization(grid_tool):
    assert grid_tool.ICON == "sketch-grid-symbolic"
    assert grid_tool.LABEL is not None
    assert "gg" in grid_tool.SHORTCUTS


@pytest.mark.ui
def test_grid_tool_is_available_no_target(grid_tool):
    assert grid_tool.is_available(None, None) is True


@pytest.mark.ui
def test_grid_tool_is_available_with_target(grid_tool):
    mock_entity = MagicMock()
    assert grid_tool.is_available(mock_entity, "entity") is False


@pytest.mark.ui
def test_grid_tool_on_press_returns_true(grid_tool):
    result = grid_tool.on_press(100, 200, 1)
    assert result is True


@pytest.mark.ui
def test_grid_tool_on_activate_shows_dialog(grid_tool, mock_element):
    with patch.object(grid_tool, "_show_dialog") as mock_show_dialog:
        grid_tool.on_activate()
        mock_show_dialog.assert_called_once()
        mock_element.set_tool.assert_called_once_with("select")


@pytest.mark.ui
def test_grid_tool_show_dialog_creates_command_on_create(
    grid_tool, mock_element
):
    mock_dialog = MagicMock()
    mock_rows_entry = MagicMock()
    mock_rows_entry.get_text.return_value = "4"
    mock_cols_entry = MagicMock()
    mock_cols_entry.get_text.return_value = "5"

    responses = {}

    def mock_connect(signal, callback):
        responses[signal] = callback

    mock_dialog.connect = mock_connect
    mock_dialog.present = MagicMock()

    with patch("gi.repository.Adw.MessageDialog") as MockDialog:
        with patch("gi.repository.Adw.EntryRow") as MockEntryRow:
            with patch("gi.repository.Gtk.ListBox") as MockListBox:
                MockDialog.return_value = mock_dialog
                MockEntryRow.side_effect = [mock_rows_entry, mock_cols_entry]
                MockListBox.return_value = MagicMock()

                grid_tool._show_dialog()

                assert "response" in responses
                responses["response"](mock_dialog, "create")

                mock_element.execute_command.assert_called_once()
                cmd = mock_element.execute_command.call_args[0][0]
                assert isinstance(cmd, GridCommand)
                assert cmd.rows == 4
                assert cmd.cols == 5


@pytest.mark.ui
def test_grid_tool_show_dialog_cancels_on_cancel(grid_tool, mock_element):
    mock_dialog = MagicMock()
    responses = {}

    def mock_connect(signal, callback):
        responses[signal] = callback

    mock_dialog.connect = mock_connect
    mock_dialog.present = MagicMock()

    with patch("gi.repository.Adw.MessageDialog") as MockDialog:
        with patch("gi.repository.Adw.EntryRow") as MockEntryRow:
            with patch("gi.repository.Gtk.ListBox") as MockListBox:
                MockDialog.return_value = mock_dialog
                MockEntryRow.return_value = MagicMock()
                MockListBox.return_value = MagicMock()

                grid_tool._show_dialog()

                responses["response"](mock_dialog, "cancel")

                mock_element.execute_command.assert_not_called()


@pytest.mark.ui
def test_grid_tool_show_dialog_handles_invalid_input(grid_tool, mock_element):
    mock_dialog = MagicMock()
    mock_rows_entry = MagicMock()
    mock_rows_entry.get_text.return_value = "invalid"
    mock_cols_entry = MagicMock()
    mock_cols_entry.get_text.return_value = "5"

    responses = {}

    def mock_connect(signal, callback):
        responses[signal] = callback

    mock_dialog.connect = mock_connect
    mock_dialog.present = MagicMock()

    with patch("gi.repository.Adw.MessageDialog") as MockDialog:
        with patch("gi.repository.Adw.EntryRow") as MockEntryRow:
            with patch("gi.repository.Gtk.ListBox") as MockListBox:
                MockDialog.return_value = mock_dialog
                MockEntryRow.side_effect = [mock_rows_entry, mock_cols_entry]
                MockListBox.return_value = MagicMock()

                grid_tool._show_dialog()

                responses["response"](mock_dialog, "create")

                mock_element.execute_command.assert_not_called()


@pytest.mark.ui
def test_grid_tool_show_dialog_handles_too_small_grid(grid_tool, mock_element):
    mock_dialog = MagicMock()
    mock_rows_entry = MagicMock()
    mock_rows_entry.get_text.return_value = "1"
    mock_cols_entry = MagicMock()
    mock_cols_entry.get_text.return_value = "1"

    responses = {}

    def mock_connect(signal, callback):
        responses[signal] = callback

    mock_dialog.connect = mock_connect
    mock_dialog.present = MagicMock()

    with patch("gi.repository.Adw.MessageDialog") as MockDialog:
        with patch("gi.repository.Adw.EntryRow") as MockEntryRow:
            with patch("gi.repository.Gtk.ListBox") as MockListBox:
                MockDialog.return_value = mock_dialog
                MockEntryRow.side_effect = [mock_rows_entry, mock_cols_entry]
                MockListBox.return_value = MagicMock()

                grid_tool._show_dialog()

                responses["response"](mock_dialog, "create")

                mock_element.execute_command.assert_not_called()


@pytest.mark.ui
def test_grid_tool_creates_construction_geometry_by_default(
    grid_tool, mock_element
):
    mock_dialog = MagicMock()
    mock_rows_entry = MagicMock()
    mock_rows_entry.get_text.return_value = "2"
    mock_cols_entry = MagicMock()
    mock_cols_entry.get_text.return_value = "2"

    responses = {}

    def mock_connect(signal, callback):
        responses[signal] = callback

    mock_dialog.connect = mock_connect
    mock_dialog.present = MagicMock()

    with patch("gi.repository.Adw.MessageDialog") as MockDialog:
        with patch("gi.repository.Adw.EntryRow") as MockEntryRow:
            with patch("gi.repository.Gtk.ListBox") as MockListBox:
                MockDialog.return_value = mock_dialog
                MockEntryRow.side_effect = [mock_rows_entry, mock_cols_entry]
                MockListBox.return_value = MagicMock()

                grid_tool._show_dialog()

                responses["response"](mock_dialog, "create")

                cmd = mock_element.execute_command.call_args[0][0]
                assert cmd.construction is True
