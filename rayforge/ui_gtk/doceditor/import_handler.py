from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from gi.repository import Gio, Adw
from ...core.vectorization_spec import VectorizationSpec, TraceSpec
from ...doceditor.file_cmd import ImportAction
from . import file_dialogs
from .import_dialog import ImportDialog

if TYPE_CHECKING:
    from ..mainwindow import MainWindow
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


def _start_interactive_import(
    win: "MainWindow",
    editor: "DocEditor",
    file_path: Path,
    mime_type: str,
    position_mm: Optional[tuple[float, float]] = None,
):
    """Creates and presents the main interactive import dialog."""
    logger.info("Starting interactive import...")

    import_dialog = ImportDialog(
        parent=win, editor=editor, file_path=file_path, mime_type=mime_type
    )

    # Define the handler locally to capture context from its closure.
    def on_dialog_response(
        sender, *, response_id: str, spec: VectorizationSpec
    ):
        _on_import_dialog_response(
            sender,
            response_id,
            spec,
            win,
            editor,
            file_path,
            mime_type,
            position_mm,
        )

    # Use weak=False to prevent the handler from being garbage collected.
    import_dialog.response.connect(on_dialog_response, weak=False)
    import_dialog.present()


def _on_import_dialog_response(
    dialog,
    response_id: str,
    spec: VectorizationSpec,
    win: "MainWindow",
    editor: "DocEditor",
    file_path: Path,
    mime_type: str,
    position_mm: Optional[tuple[float, float]] = None,
):
    """Callback for when the interactive import dialog is closed."""
    logger.info(f"Received response '{response_id}' from ImportDialog.")
    if response_id == "import":
        logger.info(
            f"Executing final import for {file_path} with spec: {spec}"
        )
        editor.file.load_file_from_path(
            file_path, mime_type, spec, position_mm
        )
        win.item_revealer.set_reveal_child(False)


def _on_file_selected(dialog, result, user_data):
    """Callback for when the user selects a file from the dialog."""
    win, editor = user_data
    try:
        file = dialog.open_finish(result)
        if not file:
            return

        file_path = Path(file.get_path())

        # Get MIME type from Gio for accuracy
        file_info = file.query_info(
            Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
            Gio.FileQueryInfoFlags.NONE,
            None,
        )
        mime_type = (
            Gio.content_type_get_mime_type(file_info.get_content_type())
            or file_info.get_content_type()
        )

        # Ask the backend what to do with this file
        action = editor.file.analyze_import_target(file_path, mime_type)

        if action == ImportAction.INTERACTIVE_CONFIG:
            _start_interactive_import(win, editor, file_path, mime_type)
        elif action == ImportAction.DIRECT_LOAD:
            editor.file.load_file_from_path(file_path, mime_type, None)
            win.item_revealer.set_reveal_child(False)
        else:  # UNSUPPORTED
            logger.warning(
                f"Unsupported file type: {mime_type} for {file_path}"
            )
            # Optionally show an error dialog here

    except Exception:
        logger.exception("Error opening file")


def start_interactive_import(win: "MainWindow", editor: "DocEditor"):
    """
    Initiates the full interactive file import process, starting with a
    file chooser dialog.
    """
    # Now passing editor to get supported file types
    file_dialogs.show_import_dialog(
        win, editor, _on_file_selected, (win, editor)
    )


def import_file_at_position(
    win: "MainWindow",
    editor: "DocEditor",
    file_path: Path,
    mime_type: str,
    position_mm: Optional[tuple[float, float]] = None,
):
    """
    Import a file and optionally position it at specified coordinates.

    Args:
        win: MainWindow instance
        editor: DocEditor instance
        file_path: Path to file to import
        mime_type: MIME type of the file
        position_mm: Optional (x, y) tuple in world coordinates (mm)
            to center the imported item
    """
    # Ask backend for routing decision
    action = editor.file.analyze_import_target(file_path, mime_type)

    if action == ImportAction.INTERACTIVE_CONFIG:
        _start_interactive_import(
            win, editor, file_path, mime_type, position_mm
        )
    elif action == ImportAction.DIRECT_LOAD:
        editor.file.load_file_from_path(
            file_path, mime_type, None, position_mm
        )
        win.item_revealer.set_reveal_child(False)
    else:
        logger.warning(f"Unsupported file type: {mime_type} for {file_path}")


def _on_batch_trace_response(
    dialog,
    response_id: str,
    editor: "DocEditor",
    file_list: list[tuple[Path, str]],
    position_mm: tuple[float, float],
    win: "MainWindow",
):
    """
    Handles the user's choice from the batch tracing configuration dialog.
    """
    if response_id == "import":
        # User confirmed - execute batch import via backend
        # We extract just the paths for the backend method
        paths = [f[0] for f in file_list]
        vectorization_spec = TraceSpec()

        editor.file.execute_batch_import(
            paths, vectorization_spec, position_mm
        )
        logger.info(f"Batch import started for {len(file_list)} files")
    # else: user cancelled, do nothing


def import_multiple_files_at_position(
    win: "MainWindow",
    editor: "DocEditor",
    file_list: list[tuple[Path, str]],
    position_mm: tuple[float, float],
):
    """
    Import multiple files with a single batch configuration dialog.

    Args:
        win: MainWindow instance
        editor: DocEditor instance
        file_list: List of (file_path, mime_type) tuples
        position_mm: (x, y) tuple in world coordinates (mm)
            to center the imported items
    """
    if not file_list:
        return

    # Check if any file in the list actually requires interactive config
    needs_config = False
    for path, mime in file_list:
        if (
            editor.file.analyze_import_target(path, mime)
            == ImportAction.INTERACTIVE_CONFIG
        ):
            needs_config = True
            break

    if not needs_config:
        # If no files need config, just load them all directly
        paths = [f[0] for f in file_list]
        vectorization_spec = TraceSpec()
        editor.file.execute_batch_import(
            paths, vectorization_spec, position_mm
        )
        return

    # If configuration is needed, show the batch dialog
    file_count = len(file_list)
    file_names = ", ".join(f.name for f, _ in file_list[:3])
    if file_count > 3:
        file_names += f" and {file_count - 3} more"

    # Show batch tracing configuration dialog
    dialog = Adw.MessageDialog(
        transient_for=win,
        modal=True,
        heading=_("Batch Import {file_count} Images").format(
            file_count=file_count
        ),
        body=_(
            "Import {file_count} images:\n{file_names}\n\n"
            "All images will be traced using the default tracing settings "
            "and positioned at the drop location."
        ).format(file_count=file_count, file_names=file_names),
    )
    dialog.add_response("cancel", _("Cancel"))
    dialog.add_response("import", _("Import All"))
    dialog.set_default_response("import")
    dialog.set_close_response("cancel")

    dialog.connect(
        "response",
        _on_batch_trace_response,
        editor,
        file_list,
        position_mm,
        win,
    )
    dialog.present()

    # Hide properties widget
    win.item_revealer.set_reveal_child(False)
