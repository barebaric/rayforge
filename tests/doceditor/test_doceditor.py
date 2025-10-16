# flake8: noqa: E402
import asyncio
import logging
from pathlib import Path
from functools import partial
import pytest
import pytest_asyncio
import re
from rayforge.shared.tasker.manager import TaskManager
from rayforge.pipeline import steps
from rayforge.doceditor.editor import DocEditor
from rayforge.core.vectorization_config import TraceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_gcode_line(line: str) -> dict:
    """
    Parses a G-code line into a dictionary of axes and values.
    Example: "G0 X10.5 Y20.0" -> {'G': 0.0, 'X': 10.5, 'Y': 20.0}
    Ignores comments.
    """
    line = line.split(";")[0].strip()
    if not line:
        return {}
    parts = {}
    # Regex to find all letter-number pairs
    tokens = re.findall(r"([A-Za-z])([-+]?\d*\.?\d+)", line)
    for command, value in tokens:
        parts[command.upper()] = float(value)
    return parts


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
    step = steps.create_contour_step(name="Vectorize", optimize=False)
    step.set_power(0.5)
    step.set_cut_speed(3000)

    # Safely access the workflow from the active layer
    workflow = editor.doc.active_layer.workflow
    assert workflow is not None, (
        "Active layer must have a workflow for this test"
    )
    workflow.add_step(step)

    svg_path = assets_path / "10x10_square.svg"
    expected_gcode_path = assets_path / "expected_square.gcode"
    output_gcode_path = tmp_path / "output.gcode"

    # --- 2. ACT ---

    # Action 1: Import the file and await its completion.
    logger.info(f"Importing file: {svg_path}")

    # Create a TraceConfig instance to trigger tracing behavior.
    trace_config = TraceConfig()

    await editor.import_file_from_path(
        svg_path, mime_type="image/svg+xml", vector_config=trace_config
    )
    logger.info("Import task has finished.")

    # Assert state after the import task has mutated the document
    assert len(editor.doc.all_workpieces) == 1
    # FIX: Assert against the file stem, which is the new default name.
    assert editor.doc.all_workpieces[0].name == "10x10_square"
    assert editor.is_processing, "Pipeline should be busy after import"

    # Wait 1: Await the reactive processing triggered by the load.
    # Increase timeout as real subprocesses are slower.
    logger.info("Waiting for Pipeline to process the imported file...")
    await editor.wait_until_settled(timeout=10)
    logger.info("Document has settled after import.")

    # Action 2 & Wait 2: Export the G-code and await its completion.
    logger.info(f"Exporting G-code to: {output_gcode_path}")
    await editor.export_gcode_to_path(output_gcode_path)
    logger.info("Export task has finished.")

    # --- 3. ASSERT ---
    assert output_gcode_path.exists()
    print(output_gcode_path.read_text())
    generated_lines = (
        output_gcode_path.read_text(encoding="utf-8").strip().splitlines()
    )
    expected_lines = (
        expected_gcode_path.read_text(encoding="utf-8").strip().splitlines()
    )

    assert len(generated_lines) == len(expected_lines), (
        "Generated G-code has a different number of lines than expected."
        f"\nGenerated lines ({len(generated_lines)}): {generated_lines}"
        f"\nExpected lines ({len(expected_lines)}): {expected_lines}"
    )

    for gen_line, exp_line in zip(generated_lines, expected_lines):
        # For non-command lines (comments, empty lines), compare directly
        if not gen_line.strip() or gen_line.strip().startswith(
            (";", "(", "%")
        ):
            assert gen_line.strip() == exp_line.strip()
            continue

        gen_parts = parse_gcode_line(gen_line)
        exp_parts = parse_gcode_line(exp_line)

        # Check that the same commands are present on both lines
        assert gen_parts.keys() == exp_parts.keys()

        # Compare values with a tolerance for floating point numbers
        for key in gen_parts:
            # A tolerance of 1 micron (0.001 mm) is reasonable for CNC
            assert gen_parts[key] == pytest.approx(exp_parts[key], abs=1e-2)
