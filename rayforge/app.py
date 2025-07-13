import warnings
import logging
import mimetypes
import argparse
import sys
import os
import gettext

# Suppress NumPy longdouble UserWarning when run under mingw on Windows
warnings.filterwarnings(
    "ignore",
    message="Signature.*for <class 'numpy.longdouble'> does not"
    " match any known type",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# When running in a PyInstaller bundle, we need to set the GI_TYPELIB_PATH
# environment variable to point to the bundled typelib files.
if hasattr(sys, '_MEIPASS'):
    os.environ['GI_TYPELIB_PATH'] = os.path.join(
        sys._MEIPASS, 'gi', 'repository'  # type: ignore
    )

import gi  # noqa: E402

# Gettext must be initialized before importing app modules.
# For installed applications, locale files are typically in
# /usr/share/locale or /usr/local/share/locale.
# For development, they are in rayforge/locale.
# gettext.install will handle finding the correct path.
APP_NAME = "rayforge"
if hasattr(sys, '_MEIPASS'):
    # In a PyInstaller bundle, locale files are in the 'rayforge/locale'
    # directory
    base_dir = sys._MEIPASS  # type: ignore
    LOCALE_DIR = os.path.join(base_dir, 'rayforge', 'locale')
else:
    # In development, they are in rayforge/locale
    LOCALE_DIR = os.path.join(os.path.dirname(__file__), "locale")

try:
    # Debug: Log the locale directory and its contents
    if os.path.exists(LOCALE_DIR):
        logging.info(f"Using translations from {LOCALE_DIR}")
        lang_folders = [
            d for d in os.listdir(LOCALE_DIR)
            if os.path.isdir(os.path.join(LOCALE_DIR, d))
        ]
        logging.info(f"{len(lang_folders)} language folders found.")

        mo_files = []
        for lang in lang_folders:
            mo_path = os.path.join(
                LOCALE_DIR, lang, "LC_MESSAGES", f"{APP_NAME}.mo"
            )
            if os.path.isfile(mo_path):
                mo_files.append(mo_path)
        logging.info(f"{len(mo_files)} .mo files found.")

        logging.info(f"Folders: {lang_folders}")
    else:
        logging.warning(f"Locale directory not found at {LOCALE_DIR}")

    # Attempt to find the translation file
    mo_file = gettext.find(APP_NAME, LOCALE_DIR)
    trans = gettext.translation(APP_NAME, LOCALE_DIR)
except Exception as e:
    logging.getLogger(__name__).warning(f"Translation setup failed: {e}")

# Make "_" available in all modules
gettext.install(APP_NAME, LOCALE_DIR)

gi.require_version('Adw', '1')
gi.require_version('Gtk', '4.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Adw  # noqa: E402
from rayforge.widgets.mainwindow import MainWindow  # noqa: E402
from rayforge.task import task_mgr  # noqa: E402
from rayforge.config import config_mgr  # noqa: E402


class App(Adw.Application):
    def __init__(self, args):
        super().__init__(application_id='com.barebaric.rayforge')
        self.set_accels_for_action("win.quit", ["<Ctrl>Q"])
        self.args = args

    def do_activate(self):
        win = MainWindow(application=self)
        if self.args.filename:
            mime_type, _ = mimetypes.guess_type(self.args.filename)
            win.load_file(self.args.filename, mime_type)
        if self.args.dumpsurface:
            win.doc.save_bitmap(self.args.dumpsurface, 10, 10)

        win.present()


def main():
    parser = argparse.ArgumentParser(
        description=_("A GCode generator for laser cutters.")
    )
    parser.add_argument(
        "filename",
        help=_("Path to the input SVG or image file."),
        nargs='?'
    )
    parser.add_argument(
        "--dumpsurface",
        metavar="FILENAME",
        help=_("Stores the work surface (no paths) as a PNG image."),
        nargs='?'
    )
    parser.add_argument(
        '--loglevel',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help=_('Set the logging level (default: INFO)')
    )

    args = parser.parse_args()

    # Set logging level based on the command-line argument
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.getLogger().setLevel(log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Application starting with log level {args.loglevel.upper()}")

    app = App(args)
    app.run(None)
    task_mgr.shutdown()
    config_mgr.save()


if __name__ == "__main__":
    main()
