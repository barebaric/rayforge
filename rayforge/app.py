# flake8: noqa: E402
import argparse
import asyncio
import logging
import mimetypes
import os
import sys
import traceback
import warnings
import gettext
import locale
from pathlib import Path
from typing import cast
from gettext import gettext as _

# Parse --config early before any rayforge imports, as they may
# import config.py which computes CONFIG_DIR at module load time
for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        os.environ["RAYFORGE_CONFIG_DIR"] = sys.argv[i + 1]
        break

from rayforge.logging_setup import setup_logging

# ===================================================================
# SECTION 1: SAFE, MODULE-LEVEL SETUP
# This code will run for the main app AND all subprocesses.
# ===================================================================

logger = logging.getLogger(__name__)


# Suppress NumPy longdouble UserWarning when run under mingw on Windows
warnings.filterwarnings(
    "ignore",
    message="Signature.*for <class 'numpy.longdouble'> does not"
    " match any known type",
)

# Gettext MUST be initialized before importing app modules.
if hasattr(sys, "_MEIPASS"):
    # In a PyInstaller bundle, the project root is in a temporary
    # directory stored in sys._MEIPASS.
    base_dir = Path(sys._MEIPASS)  # type: ignore
else:
    base_dir = Path(__file__).parent.parent

# Set the locale from environment so Python's locale module (e.g.
# locale.format_string) respects LC_NUMERIC for decimal separators.
try:
    locale.setlocale(locale.LC_ALL, "")
except locale.Error:
    pass

# Configure gettext with the locale directory
locale_dir = base_dir / "rayforge" / "locale"
gettext.bindtextdomain("rayforge", str(locale_dir))
gettext.textdomain("rayforge")

# --------------------------------------------------------
# GObject Introspection Repository (gi)
# --------------------------------------------------------
# When running in a PyInstaller bundle, we need to set the GI_TYPELIB_PATH
# environment variable to point to the bundled typelib files.
if hasattr(sys, "_MEIPASS"):
    if sys.platform == "darwin":
        # macOS PyInstaller bundles use a Frameworks directory structure
        # that requires specific environment variables for dynamic linking
        # and GObject Introspection to work correctly.
        frameworks_dir = Path(sys._MEIPASS).parent / "Frameworks"
        bundled_typelibs = frameworks_dir / "gi_typelibs"
        bundled_gio_modules = frameworks_dir / "gio_modules"
        lib_path = str(frameworks_dir)
        # DYLD_LIBRARY_PATH: Directories for dynamic linker to search
        existing_dyld = os.environ.get("DYLD_LIBRARY_PATH")
        os.environ["DYLD_LIBRARY_PATH"] = (
            lib_path if not existing_dyld else f"{lib_path}:{existing_dyld}"
        )
        # DYLD_FALLBACK_LIBRARY_PATH: Fallback if DYLD_LIBRARY_PATH fails
        os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", lib_path)
        # GI_TYPELIB_PATH: Path to GObject Introspection typelib files
        seen = set()
        candidates = []
        for path in [bundled_typelibs]:
            if path.exists():
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(resolved)
        if candidates:
            os.environ["GI_TYPELIB_PATH"] = ":".join(candidates)
            logger.info(f"GI_TYPELIB_PATH is {os.environ['GI_TYPELIB_PATH']}")
        else:
            logger.warning("No GI typelibs found for bundled build.")
        # GIO_EXTRA_MODULES: Path to additional GIO modules
        if bundled_gio_modules.exists():
            os.environ.setdefault(
                "GIO_EXTRA_MODULES", str(bundled_gio_modules)
            )
    else:
        # Non-macOS platforms use the standard gi/repository structure
        typelib_path = base_dir / "gi" / "repository"
        logger.info(f"GI_TYPELIB_PATH is {typelib_path}")
        os.environ["GI_TYPELIB_PATH"] = str(typelib_path)
        files = [p.name for p in typelib_path.iterdir()]
        logger.info(f"Files in typelib path: {files}")

        # On Windows, subprocesses need explicit DLL search path.
        # This must be at module level to run during worker import.
        if sys.platform == "win32":
            logger.info(
                f"Windows build detected. Adding '{base_dir}' to DLL search path."
            )
            try:
                os.add_dll_directory(str(base_dir))
            except OSError:
                pass


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    Catches unhandled exceptions, logs them, and shows a user-friendly dialog.
    This is crucial for --noconsole builds.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Print full traceback to stderr (console or log)
    traceback.print_exception(exc_type, exc_value, exc_traceback)

    logger.error(
        "Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback)
    )
    logging.shutdown()


def main():
    # ===================================================================
    # SECTION 2: MAIN APPLICATION ENTRY POINT
    # This function contains all logic that should ONLY run in the
    # main process.
    # ===================================================================

    # Set the global exception handler.
    sys.excepthook = handle_exception

    # We need Adw for the class definition, so this one import is okay here.
    import gi

    gi.require_version("Adw", "1")
    from gi.repository import Adw, Gio, GLib, Gtk

    if os.environ.get("SNAP"):
        settings = Gtk.Settings.get_default()
        if settings:
            settings.set_property("gtk-icon-theme-name", "Adwaita")

    from rayforge.context import get_context

    class App(Adw.Application):
        def __init__(self, args):
            super().__init__(application_id="org.rayforge.rayforge")
            from rayforge.ui_gtk.shared.keyboard import PRIMARY_ACCEL

            self.args = args
            self.win = None
            self._register_app_actions()
            self.set_accels_for_action("app.quit", [f"{PRIMARY_ACCEL}q"])
            self.set_accels_for_action(
                "app.preferences", [f"{PRIMARY_ACCEL}comma"]
            )

        def _register_app_actions(self):
            action_specs = (
                ("about", self._on_app_about),
                ("preferences", self._on_app_preferences),
                ("quit", self._on_app_quit),
            )
            for name, callback in action_specs:
                action = Gio.SimpleAction.new(name, None)
                action.connect("activate", callback)
                self.add_action(action)

        def _get_main_window(self):
            from rayforge.ui_gtk.mainwindow import MainWindow

            window = self.get_active_window()
            if isinstance(window, MainWindow):
                return window
            if isinstance(self.win, MainWindow):
                return self.win
            return None

        def _on_app_about(self, action, param):
            window = self._get_main_window()
            if window is None:
                return
            window.show_about_dialog(None, None)

        def _on_app_preferences(self, action, param):
            window = self._get_main_window()
            if window is None:
                return
            window.show_settings(None, None)

        def _on_app_quit(self, action, param):
            logger.debug("_on_app_quit called.")
            window = self._get_main_window()
            if window is None:
                self.quit()
                return
            window.on_quit_action(None, None)

        def do_shutdown(self):
            logger.info("App.do_shutdown called.")
            Adw.Application.do_shutdown(self)
            logger.info("App.do_shutdown completed. Calling self.quit().")
            self.quit()

        def do_activate(self):
            if self.win is not None:
                self.win.present()
                return

            # Import the window here to avoid module-level side-effects
            from rayforge.ui_gtk.mainwindow import MainWindow

            self.win = MainWindow(application=self)

            # Don't load files until the window is fully mapped and
            # allocated on screen. The 'map' signal guarantees this.
            if self.args.filenames:
                # We connect a one-shot handler to the 'map' event.
                self.win.connect("map", self._load_initial_files)
            else:
                # No files specified on command line, check config for
                # startup behavior
                self.win.connect("map", self._load_startup_files)

            if self.args.uiscript:
                self.win.connect("map", self._run_uiscript)

            self.win.present()

            # Now that the UI is active, trigger the initial machine connections.
            context = get_context()
            if context.machine_mgr:
                context.machine_mgr.initialize_connections()

        def _load_initial_files(self, widget):
            """
            Loads files passed via the command line. This is called from the
            'map' signal handler to ensure the main window is fully initialized.
            Command line files always override the startup behavior setting.
            """
            # These imports must be inside the method.
            from rayforge.core.vectorization_spec import (
                TraceSpec,
                PassthroughSpec,
            )
            from rayforge.image import ImporterFeature

            assert self.win is not None
            editor = self.win.doc_editor

            # self.args.filenames will be a list of paths
            for filename in self.args.filenames:
                file_path = Path(filename)

                if file_path.suffix.lower() == ".ryp":
                    self.win.load_project(file_path)
                    continue

                mime_type, __ = mimetypes.guess_type(file_path)

                importer_cls, features = editor.file.get_importer_info(
                    file_path, mime_type
                )
                if not importer_cls:
                    logger.warning(
                        f"No importer found for '{file_path.name}'. Skipping."
                    )

                    editor.notification_requested.send(
                        self,
                        message=_(
                            "Cannot open '{file}'. The required addon may be disabled."
                        ).format(file=file_path.name),
                    )
                    continue

                vectorization_spec = None
                if self.args.trace:
                    if ImporterFeature.BITMAP_TRACING not in features:
                        logger.error(
                            f"Error: The importer for '{file_path.name}' does "
                            "not support tracing."
                        )
                        sys.exit(1)
                    vectorization_spec = TraceSpec()
                elif self.args.vector:
                    if ImporterFeature.DIRECT_VECTOR not in features:
                        logger.warning(
                            f"Warning: The importer for '{file_path.name}' "
                            "may not support direct vector import."
                        )
                    vectorization_spec = PassthroughSpec()

                # If no flag is passed, vectorization_spec remains None,
                # allowing the importer to use its smart default.
                editor.file.load_file_from_path(
                    filename=file_path,
                    mime_type=mime_type,
                    vectorization_spec=vectorization_spec,
                )

            if self.args.exit:
                get_context().exit_after_settle = True
                editor.document_settled.connect(self._on_document_settled_exit)

            return GLib.SOURCE_REMOVE

        def _on_document_settled_exit(self, sender):
            ctx = get_context()
            if ctx.exit_pending:
                return
            ctx.exit_pending = True
            assert self.win is not None
            if self.win.doc_editor.is_processing:
                ctx.exit_pending = False
                return
            logger.info("Document settled, exiting due to --exit flag.")
            self.quit()

        def quit_idle(self):
            """Thread-safe quit for use from background threads."""
            GLib.idle_add(self.quit)

        def _run_uiscript(self, widget):
            """Schedule UI script execution after window is mapped."""
            from rayforge.uiscript import run_script

            run_script(Path(self.args.uiscript), self, self.win)
            return GLib.SOURCE_REMOVE

        def _load_startup_files(self, widget):
            """
            Loads files based on the startup behavior setting when no files
            are specified on the command line.
            """
            from rayforge.core.config import StartupBehavior

            assert self.win is not None
            context = get_context()
            config = context.config

            startup_behavior = config.startup_behavior
            project_path = None

            if startup_behavior == StartupBehavior.LAST_PROJECT.value:
                project_path = config.last_opened_project
            elif startup_behavior == StartupBehavior.SPECIFIC_PROJECT.value:
                project_path = config.startup_project_path

            if project_path and project_path.exists():
                if project_path.suffix.lower() == ".ryp":
                    logger.info(f"Loading startup project from {project_path}")
                    self.win.load_project(project_path)
                else:
                    logger.warning(
                        f"Startup project path {project_path} is not a .ryp file"
                    )
            elif project_path:
                logger.warning(
                    f"Startup project path {project_path} does not exist"
                )

            if self.args.exit:
                get_context().exit_after_settle = True
                self.win.doc_editor.document_settled.connect(
                    self._on_document_settled_exit
                )

            return GLib.SOURCE_REMOVE

    # Import version for the --version flag.
    from rayforge import __version__

    parser = argparse.ArgumentParser(
        description=_("A GCode generator for laser cutters.")
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "filenames",
        help=_("Paths to one or more input SVG or image files."),
        nargs="*",
    )

    # Create a mutually exclusive group for import mode flags
    import_mode_group = parser.add_mutually_exclusive_group()
    import_mode_group.add_argument(
        "--vector",
        action="store_true",
        help=_(
            "Force import as direct vectors. This is the default for "
            "supported files."
        ),
    )
    import_mode_group.add_argument(
        "--trace",
        action="store_true",
        help=_(
            "Force import by tracing the file's bitmap representation. "
            "Aborts if not supported."
        ),
    )

    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help=_("Set the logging level (default: INFO)"),
    )

    parser.add_argument(
        "--exit",
        action="store_true",
        help=_(
            "Exit after importing documents and the editor has settled. "
            "Useful for testing."
        ),
    )

    parser.add_argument(
        "--uiscript",
        metavar="SCRIPT",
        help=_(
            "Path to a Python script to execute after the main window "
            "is fully loaded. Useful for automation and testing."
        ),
    )

    parser.add_argument(
        "--config",
        metavar="DIR",
        help=_(
            "Path to a custom configuration directory. "
            "Useful for testing with isolated configs."
        ),
    )

    args = parser.parse_args()

    # Set logging level based on the command-line argument.
    setup_logging(args.loglevel)
    logger.info(f"Application starting with log level {args.loglevel.upper()}")

    # ===================================================================
    # SECTION 3: PLATFORM SPECIFIC INITIALIZATION
    # ===================================================================

    # Set the PyOpenGL platform before importing anything that uses OpenGL.
    # 'egl' is generally the best choice for GTK4 on modern Linux (Wayland/X11).
    # On Windows and macOS, letting PyOpenGL auto-detect is more reliable.
    if sys.platform.startswith("linux"):
        logger.info("Linux detected. Setting PYOPENGL_PLATFORM=egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Print PyCairo version
    import cairo

    logger.info(f"PyCairo version: {cairo.version}")

    # Register the standalone 'cairo' module
    # as a foreign type *before* the GObject-introspected cairo is loaded.
    gi.require_foreign("cairo")

    # Now, when gi.repository.cairo is loaded, it will know how to
    # interact with the already-imported standalone module.
    gi.require_version("cairo", "1.0")
    gi.require_version("Pango", "1.0")
    gi.require_version("PangoCairo", "1.0")
    gi.require_version("Gtk", "4.0")
    gi.require_version("GdkPixbuf", "2.0")

    # Initialize the 3D canvas module to check for OpenGL availability.
    # This must be done after setting the platform env var and after
    # making Gtk available in gi, as the canvas uses Gtk.
    # The rest of the app can now check
    # `rayforge.ui_gtk.sim3d.initialized`.
    # It is safe to import other modules that depend on canvas3d after this.
    from rayforge.ui_gtk.sim3d import initialize

    initialize()

    # Import modules that depend on GTK or manage global state
    import rayforge.shared.tasker
    from rayforge.shared.tasker.manager import TaskManagerProxy
    from rayforge.shared.util.glib import idle_add
    from rayforge.worker_init import initialize_worker

    # Get the context first to ensure the ArtifactStore is created
    # before the TaskManager is initialized. This breaks the circular
    # dependency chain (app -> config -> machine -> task_manager).
    get_context()

    # Initialize the TaskManager with the worker initializer.
    # This MUST happen before accessing addon_mgr because the
    # MachineManager creates machines which import task_mgr, which
    # would trigger the creation of the TaskManager.
    task_mgr_proxy = cast(TaskManagerProxy, rayforge.shared.tasker.task_mgr)
    task_mgr_proxy.initialize(
        worker_initializer=initialize_worker,
        main_thread_scheduler=idle_add,
    )

    # Wire up addon module paths from WorkerPoolManager to AddonManager.
    # This allows workers to resolve rayforge_addons.* modules during
    # unpickling without loading all addons at startup.
    shared_state = task_mgr_proxy.get_shared_state()
    get_context().addon_mgr.set_task_manager(task_mgr_proxy)
    get_context().addon_mgr.set_shared_state(shared_state)

    # Run application
    app = App(args)
    exit_code = app.run(None)
    logger.info("app.run() returned with exit_code=%s", exit_code)
    if app.win is None:
        logger.info(
            "No window created (another instance is likely running). Exiting."
        )
        return exit_code

    # ===================================================================
    # SECTION 4: SHUTDOWN SEQUENCE
    # ===================================================================

    logger.info("Application exiting.")
    context = get_context()

    # 1. Define an async function to shut down high-level components.
    async def shutdown_async():
        logger.info("Starting graceful async shutdown...")
        # The context now handles shutting down all its owned managers
        # (machine_mgr, camera_mgr, artifact_store) in the correct order.
        await context.shutdown()
        logger.info("Async shutdown complete.")

    # 2. Run the async shutdown on the TaskManager's event loop and wait for it.
    loop = rayforge.shared.tasker.task_mgr.loop
    if loop.is_running():
        logger.info(f"Running async shutdown on loop {loop}...")
        future = asyncio.run_coroutine_threadsafe(shutdown_async(), loop)
        try:
            # Block until the async cleanup is finished.
            future.result(timeout=10)
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
    else:
        logger.warning(
            "Task manager loop not running, skipping async shutdown."
        )

    # 3. Save configuration. This happens AFTER async tasks are done.
    logger.info("Saving configuration")
    if context.config_mgr:
        context.config_mgr.save()
        logger.info("Saved config.")
    else:
        logger.info("No config manager to save.")

    # 4. As the final step, clean up the document editor,
    # and shut down the task manager itself.
    # The context shutdown (including artifact store) now happens in the async
    # part above, so we only need to clean up the editor here.
    logger.info("Cleaning up DocEditor")
    app.win.doc_editor.cleanup()
    logger.info("DocEditor cleaned up.")

    logger.info("Shutting down TaskManager")
    rayforge.shared.tasker.task_mgr.shutdown()
    logger.info("Task manager shut down.")

    return exit_code


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()  # needed to use multiprocessing in PyInstaller bundles
    sys.exit(main())
