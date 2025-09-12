# flake8: noqa: E402
import asyncio
import logging
from pathlib import Path
from functools import partial
import pytest
import pytest_asyncio
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_config_manager(tmp_path):
    """Provides a test-isolated ConfigManager."""
    from rayforge import config

    # We still need to set up the config globals for modules that might
    # use them (like WorkPiece), but the DocEditor itself will get an
    # explicit instance.
    temp_config_dir = tmp_path / "config"
    temp_machine_dir = temp_config_dir / "machines"
    config.CONFIG_DIR = temp_config_dir
    config.MACHINE_DIR = temp_machine_dir

    config.initialize_managers()
    yield config.config_mgr
    # Reset globals after test
    config.config = None
    config.config_mgr = None
    config.machine_mgr = None


@pytest_asyncio.fixture
async def task_mgr():
    """
    Provides a test-isolated TaskManager, configured to bridge its main-thread
    callbacks to the asyncio event loop.
    """
    from rayforge.shared.tasker.manager import TaskManager

    main_loop = asyncio.get_running_loop()

    def asyncio_scheduler(callback, *args, **kwargs):
        main_loop.call_soon_threadsafe(partial(callback, *args, **kwargs))

    # Instantiate the TaskManager with our custom scheduler, eliminating the
    # need for any monkeypatching.
    tm = TaskManager(main_thread_scheduler=asyncio_scheduler)

    yield tm

    tm.shutdown()


@pytest_asyncio.fixture
async def editor(task_mgr, test_config_manager):
    """
    Provides a fully configured DocEditor.
    """
    from rayforge.doceditor.editor import DocEditor

    return DocEditor(task_manager=task_mgr, config_manager=test_config_manager)


@pytest.fixture
def assets_path() -> Path:
    """
    Fixture providing the path to this test's assets directory.
    """
    path = Path(__file__).parent / "assets"
    if not path.exists():
        pytest.fail(f"Asset directory not found. Please create it at: {path}")
    return path


@pytest.mark.asyncio
async def test_import_svg_export_gcode(editor, tmp_path, assets_path):
    """Full end-to-end test using a real subprocess for ops generation."""
    # --- 1. ARRANGE ---
    from rayforge.pipeline import steps

    step = steps.create_contour_step(name="Vectorize")
    step.set_power(500)
    step.set_cut_speed(3000)
    editor.doc.active_layer.workflow.add_step(step)

    svg_path = assets_path / "10x10_square.svg"
    expected_gcode_path = assets_path / "expected_square.gcode"
    output_gcode_path = tmp_path / "output.gcode"

    # --- 2. ACT ---

    # Action 1: Load the file (this is a synchronous model mutation)
    logger.info(f"Importing file: {svg_path}")
    editor.file.load_file_from_path(svg_path, mime_type="image/svg+xml")

    # Assert initial state after synchronous load
    assert len(editor.doc.all_workpieces) == 1
    assert editor.doc.all_workpieces[0].name == "10x10_square.svg"
    assert editor.is_processing

    # Wait 1: Await the reactive processing triggered by the load.
    # Increase timeout as real subprocesses are slower.
    logger.info("Waiting for OpsGenerator to process the imported file...")
    await editor.wait_until_settled(timeout=10)
    logger.info("Document has settled after import.")

    # Action 2 & Wait 2: Export the G-code and await its completion.
    logger.info(f"Exporting G-code to: {output_gcode_path}")
    await editor.export_gcode_to_path(output_gcode_path)
    logger.info("Export task has finished.")

    # --- 3. ASSERT ---
    assert output_gcode_path.exists()
    generated_gcode = output_gcode_path.read_text(encoding="utf-8").strip()
    expected_gcode = expected_gcode_path.read_text(encoding="utf-8").strip()
    assert generated_gcode == expected_gcode