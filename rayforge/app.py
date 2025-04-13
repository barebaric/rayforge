import logging
import mimetypes
import argparse
import gi

gi.require_version('Adw', '1')
gi.require_version('Gtk', '4.0')
from gi.repository import Adw  # noqa: E402
from .widgets.mainwindow import MainWindow  # noqa: E402
from .task import task_mgr  # noqa: E402
from .config import config_mgr  # noqa: E402


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
        description="A GCode generator for laser cutters."
    )
    parser.add_argument(
        "filename",
        help="Path to the input SVG or image file.",
        nargs='?'
    )
    parser.add_argument(
        "--dumpsurface",
        metavar="FILENAME",
        help="Stores the work surface (no paths) as a PNG image.",
        nargs='?'
    )
    parser.add_argument(
        '--loglevel',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Configure logging based on the command-line argument
    log_level = getattr(logging, args.loglevel.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Application starting with log level {args.loglevel.upper()}")

    app = App(args)
    app.run(None)
    task_mgr.shutdown()
    config_mgr.save()


if __name__ == "__main__":
    main()
