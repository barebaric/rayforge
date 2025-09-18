from __future__ import annotations
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from gi.repository import Gio, Adw

from ...core.vectorization_config import TraceConfig
from . import file_dialogs

if TYPE_CHECKING:
    from ...mainwindow import MainWindow
    from ..editor import DocEditor

logger = logging.getLogger(__name__)


def _on_svg_options_response(
    dialog,
    response_id: str,
    editor: "DocEditor",
    file_path: Path,
    mime_type: str,
    win: "MainWindow",
):
    """Handles the user's choice from the SVG import dialog."""
    vector_config: Optional[TraceConfig] = None
    if response_id == "trace":
        vector_config = TraceConfig()
    elif response_id == "direct":
        vector_config = None  # None now signifies direct vector import
    else:  # "cancel" or the dialog was closed
        return

    editor.file.load_file_from_path(file_path, mime_type, vector_config)
    # Hide properties widget in case something was selected before import
    win.item_revealer.set_reveal_child(False)


def _show_svg_import_dialog(
    win: "MainWindow", editor: "DocEditor", file_path: Path, mime_type: str
):
    """Shows a dialog asking the user how to import an SVG."""
    dialog = Adw.MessageDialog(
        transient_for=win,
        modal=True,
        heading=_("SVG Import Options"),
        body=_(
            "How would you like to import this SVG file?\n\n"
            "<b>Import Vectors Directly:</b> High-fidelity, but may not "
            " support all SVG features.\n"
            "<b>Trace Bitmap:</b> More robust, but converts the SVG to "
            "pixels first, which may lose detail."
        ),
        body_use_markup=True,
    )
    dialog.add_response("direct", _("Import Vectors Directly"))
    dialog.add_response("trace", _("Trace Bitmap"))
    dialog.set_default_response("direct")
    dialog.set_close_response("cancel")

    dialog.connect(
        "response", _on_svg_options_response, editor, file_path, mime_type, win
    )
    dialog.present()


def _on_file_selected(dialog, result, user_data):
    """Callback for when the user selects a file from the dialog."""
    win, editor = user_data
    try:
        file = dialog.open_finish(result)
        if not file:
            return

        file_path = Path(file.get_path())
        file_info = file.query_info(
            Gio.FILE_ATTRIBUTE_STANDARD_CONTENT_TYPE,
            Gio.FileQueryInfoFlags.NONE,
            None,
        )
        mime_type = file_info.get_content_type()

        # For SVGs, ask the user how they want to import it.
        if mime_type == "image/svg+xml":
            _show_svg_import_dialog(win, editor, file_path, mime_type)
        else:
            # For all other raster types, default to tracing.
            vector_config = TraceConfig()
            editor.file.load_file_from_path(
                file_path, mime_type, vector_config
            )
            # Hide properties widget in case something was selected before
            # import
            win.item_revealer.set_reveal_child(False)

    except Exception:
        logger.exception("Error opening file")


def start_interactive_import(win: "MainWindow", editor: "DocEditor"):
    """
    Initiates the full interactive file import process, starting with a
    file chooser dialog.
    """
    file_dialogs.show_import_dialog(win, _on_file_selected, (win, editor))
