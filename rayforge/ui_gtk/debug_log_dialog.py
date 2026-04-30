import logging
from pathlib import Path
from typing import Optional, Callable, TYPE_CHECKING
from gettext import gettext as _

from gi.repository import Adw, Gio, GLib, Gtk

from ..context import get_context
from ..debug import DebugDumpManager

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class DebugLogDialog(Adw.MessageDialog):
    """
    Dialog shown before creating a debug dump archive.
    Lets the user choose whether to include the current
    project in the archive.
    """

    def __init__(
        self,
        parent: Gtk.Window,
        editor: Optional["DocEditor"] = None,
        on_saved: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        super().__init__(transient_for=parent)
        self._editor = editor
        self._parent = parent
        self._on_saved = on_saved
        self._on_error = on_error

        self.set_heading(_("Save Debug Log"))
        self.set_body(
            _(
                "Create a ZIP archive with log files and system "
                "information for troubleshooting."
            ),
        )

        self._include_switch = Adw.SwitchRow(
            title=_("Include current project"),
            subtitle=_("Add the current project file to the debug archive"),
        )
        self._include_switch.set_active(True)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_top(12)
        content_box.append(self._include_switch)
        self.set_extra_child(content_box)

        self.add_response("cancel", _("Cancel"))
        self.add_response("save", _("_Save"))
        self.set_default_response("save")
        self.set_close_response("cancel")
        self.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)

        self.connect("response", self._on_response)

    @property
    def include_project(self) -> bool:
        return self._include_switch.get_active()

    def _create_archive(self) -> Optional[Path]:
        editor = self._editor if self.include_project else None
        return get_context().debug_dump_manager.create_dump_archive(
            editor=editor,
        )

    def _on_response(self, dialog, response_id):
        self.destroy()
        if response_id != "save":
            return

        archive_path = self._create_archive()

        if not archive_path:
            if self._on_error:
                self._on_error(_("Failed to create debug archive."))
            return

        file_dialog = Gtk.FileDialog.new()
        file_dialog.set_title(_("Save Debug Log"))
        file_dialog.set_initial_name(archive_path.name)

        def save_callback(file_dialog, result):
            try:
                destination_file = file_dialog.save_finish(result)
                if destination_file:
                    destination_path = Path(destination_file.get_path())
                    DebugDumpManager.save_archive_to(
                        archive_path, destination_path
                    )
                    if self._on_saved:
                        self._on_saved(destination_path.name)
                    return
            except GLib.Error as e:
                if not e.matches(
                    Gio.io_error_quark(),
                    Gio.IOErrorEnum.CANCELLED,
                ):
                    if self._on_error:
                        self._on_error(
                            _("Error saving file: {msg}").format(msg=e.message)
                        )
            except Exception as e:
                if self._on_error:
                    self._on_error(
                        _("An unexpected error occurred: {error}").format(
                            error=e
                        )
                    )
            finally:
                if archive_path.exists():
                    archive_path.unlink()

        file_dialog.save(self._parent, None, save_callback)
