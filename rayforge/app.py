# flake8: noqa: E402
import warnings
import logging
import mimetypes
import argparse
import sys
import os
import gettext
from pathlib import Path

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

# --------------------------------------------------------
# Gettext MUST be initialized before importing app modules
# --------------------------------------------------------
if hasattr(sys, '_MEIPASS'):
    # In a PyInstaller bundle, the project root is in a temporary
    # directory stored in sys._MEIPASS.
    base_dir = Path(sys._MEIPASS)  # type: ignore
else:
    # In other environments, this is safer.
    base_dir = Path(__file__).parent.parent

# Make "_" available in all modules
locale_dir = base_dir / 'rayforge' / 'locale'
logging.info(f"Loading locales from {locale_dir}")
gettext.install("rayforge", locale_dir)

# --------------------------------------------------------
# GObject Introspection Repository (gi)
# --------------------------------------------------------
# When running in a PyInstaller bundle, we need to set the GI_TYPELIB_PATH
# environment variable to point to the bundled typelib files.
if hasattr(sys, '_MEIPASS'):
    typelib_path = base_dir / 'gi' / 'repository'
    logging.info(f"GI_TYPELIB_PATH is {typelib_path}")
    os.environ['GI_TYPELIB_PATH'] = str(typelib_path)
    files = [p.name for p in typelib_path.iterdir()]
    logging.info(f"Files in typelib path: {files}")

# --------------------------------------------------------
# Test PyCairo functionality
# --------------------------------------------------------
import cairo
logging.info(f"PyCairo version: {cairo.version}")
# Create a dummy surface to test PyCairo
surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
ctx = cairo.Context(surface)
logging.info("Successfully created cairo.Context")

# --------------------------------------------------------
# Now we should be ready to import the app.
# --------------------------------------------------------
import gi
gi.require_version('Adw', '1')
gi.require_version('Gtk', '4.0')
gi.require_version('GdkPixbuf', '2.0')
from gi.repository import Adw
from rayforge.widgets.mainwindow import MainWindow
from rayforge.task import task_mgr
from rayforge.config import config_mgr


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
