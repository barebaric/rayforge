import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from gi.repository import Adw, Gio, GLib, Gtk

from .. import __version__, const
from ..context import get_context
from .doceditor import file_dialogs

if TYPE_CHECKING:
    from .mainwindow import MainWindow
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class ProjectCmd:
    """Handles project file operations (new, open, save, recent files)."""

    def __init__(self, win: "MainWindow", editor: "DocEditor"):
        self._win = win
        self._editor = editor

    def show_unsaved_changes_dialog(self, callback: Callable[[str], None]):
        """
        Shows a dialog asking the user what to do with unsaved changes.
        The callback will be called with the response: 'save', 'discard',
        or 'cancel'.
        """
        dialog = Adw.MessageDialog(
            transient_for=self._win,
            heading=_("Unsaved Changes"),
            body=_(
                "The current project has unsaved changes. "
                "Do you want to save them?"
            ),
        )
        dialog.add_response("cancel", _("_Cancel"))
        dialog.add_response("discard", _("_Don't Save"))
        dialog.add_response("save", _("_Save"))
        dialog.set_default_response("save")
        dialog.set_close_response("cancel")
        dialog.set_response_appearance(
            "save", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.set_response_appearance(
            "discard", Adw.ResponseAppearance.DESTRUCTIVE
        )

        def on_response(d, response_id):
            d.destroy()
            callback(response_id)

        dialog.connect("response", on_response)
        dialog.present()

    def on_new_project(self, action, param):
        """Action handler for creating a new project."""
        if self._editor.is_saved:
            self._do_create_new_project()
        else:
            self.show_unsaved_changes_dialog(
                self._on_new_project_dialog_response
            )

    def _on_new_project_dialog_response(self, response: str):
        """Callback for unsaved changes dialog in on_new_project."""
        if response == "cancel":
            return
        if response == "save":
            self.on_save_project(None, None)
            if not self._editor.is_saved:
                return
        self._do_create_new_project()

    def _do_create_new_project(self):
        """Actually creates a new project."""
        from ..core.doc import Doc

        new_doc = Doc()
        self._editor.set_doc(new_doc)
        self._editor.set_file_path(None)
        self._editor.mark_as_saved()

        logger.info("Created new project")
        self._win._on_editor_notification(
            self._win, message=_("New project created")
        )

    def on_open_project(self, action, param):
        """Action handler for opening a project file."""
        if self._editor.is_saved:
            file_dialogs.show_open_project_dialog(
                self._win, self._on_open_project_response
            )
        else:
            self.show_unsaved_changes_dialog(
                self._on_open_project_dialog_response
            )

    def _on_open_project_dialog_response(self, response: str):
        """Callback for unsaved changes dialog in on_open_project."""
        if response == "cancel":
            return
        if response == "save":
            self.on_save_project(None, None)
            if not self._editor.is_saved:
                return
        file_dialogs.show_open_project_dialog(
            self._win, self._on_open_project_response
        )

    def _on_open_project_response(self, dialog, result, user_data):
        """Callback for the open project dialog."""
        try:
            file = dialog.open_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())
            self.load_project(file_path)
        except GLib.Error as e:
            logger.error(f"Error opening file: {e.message}")
            return

    def on_save_project(self, action, param):
        """Action handler for saving the current project."""
        file_path = self._editor.file_path
        if file_path:
            success = self._editor.file.save_project_to_path(file_path)
            if success:
                self._win.on_doc_changed(self._editor.doc)
                self.add_to_recent_manager(file_path)
        else:
            self.on_save_project_as(action, param)

    def on_save_project_as(self, action, param):
        """Action handler for saving the project with a new name."""
        initial_name = None
        if self._editor.file_path:
            initial_name = self._editor.file_path.name

        file_dialogs.show_save_project_dialog(
            self._win, self._on_save_project_response, initial_name
        )

    def _on_save_project_response(self, dialog, result, user_data):
        """Callback for the save project dialog."""
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())

            success = self._editor.file.save_project_to_path(file_path)
            if success:
                self._win.on_doc_changed(self._editor.doc)
                self.add_to_recent_manager(file_path)
        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
            return

    def load_project(self, file_path: Path):
        """
        Public method to load a project from a given path.
        Updates recent files and tracks the last opened project.
        """
        success = self._editor.file.load_project_from_path(file_path)
        if success:
            self._win.on_doc_changed(self._editor.doc)
            self.add_to_recent_manager(file_path)
            context = get_context()
            context.config.set_last_opened_project(file_path)

    def on_open_recent(self, action, param):
        """Action handler for opening a file from the recent menu."""
        uri = param.get_string()
        file = Gio.File.new_for_uri(uri)
        path = file.get_path()
        if path is None:
            return
        file_path = Path(path)

        if self._editor.is_saved:
            self.load_project(file_path)
        else:

            def on_response(response_id):
                self._on_open_recent_dialog_response(response_id, file_path)

            self.show_unsaved_changes_dialog(on_response)

    def _on_open_recent_dialog_response(self, response: str, file_path: Path):
        """Callback for unsaved changes dialog when opening a recent file."""
        if response == "cancel":
            return
        if response == "save":
            self.on_save_project(None, None)
            if not self._editor.is_saved:
                return
        self.load_project(file_path)

    def add_to_recent_manager(self, file_path: Path):
        """Adds a project file path to the Gtk.RecentManager with metadata."""
        uri = file_path.resolve().as_uri()
        app = self._win.get_application()
        if not app:
            logger.warning(
                "Could not get application to register recent file."
            )
            return

        app_id = app.get_application_id()
        if not app_id:
            logger.warning(
                "Application ID is not set, cannot register recent file."
            )
            return

        recent_data = Gtk.RecentData()
        recent_data.display_name = file_path.name
        recent_data.mime_type = const.MIME_TYPE_PROJECT
        recent_data.app_name = app_id
        recent_data.app_exec = f'"{sys.executable}" %f'

        self._win.recent_manager.add_full(uri, recent_data)

    def update_recent_files_menu(self, *args):
        """
        Updates the 'Open Recent' submenu based on the contents of the
        Gtk.RecentManager.
        """
        app = self._win.get_application()
        if not app:
            self._win.menu_model.update_recent_files_menu([])
            return

        app_id = app.get_application_id()
        if not app_id:
            self._win.menu_model.update_recent_files_menu([])
            return

        items = self._win.recent_manager.get_items()
        ryp_items = [
            info
            for info in items
            if info.get_uri().endswith(".ryp")
            and app_id in info.get_applications()
        ]
        self._win.menu_model.update_recent_files_menu(ryp_items)

    def on_saved_state_changed(self, sender):
        """Handles saved state changes from DocEditor."""
        self.update_window_title()
        self._win.action_manager.update_action_states()

    def update_window_title(self):
        """Updates the window title based on file name and saved state."""
        file_path = self._editor.file_path
        is_saved = self._editor.is_saved

        doc_title = file_path.name if file_path else _("Untitled")
        if is_saved:
            title = f"{doc_title} - {const.APP_NAME}"
        else:
            title = f"{doc_title}* - {const.APP_NAME}"

        subtitle = __version__ or ""

        window_title = Adw.WindowTitle(title=title, subtitle=subtitle)
        self._win.header_bar.set_title_widget(window_title)
