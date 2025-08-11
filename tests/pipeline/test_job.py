import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand, MoveToCommand
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.generator import OpsGenerator
from rayforge.shared.tasker.manager import CancelledError
from rayforge.pipeline.job import generate_job_ops
from rayforge.pipeline.steps import create_outline_step
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
from rayforge.importer import SvgImporter


@pytest.fixture(autouse=True)
def setup_job_test(mocker):
    mocker.patch("builtins._", lambda s: s, create=True)


@pytest.fixture
def machine():
    m = Machine()
    m.dimensions = (200, 150)
    m.y_axis_down = True
    m.add_head(Laser())
    return m


@pytest.fixture
def doc():
    d = Doc()
    # Start with a clean slate for tests
    d.layers[0].workflow.set_steps([])
    return d


@pytest.fixture
def mock_ops_generator():
    """A mock OpsGenerator that we can configure for tests."""
    return MagicMock(spec=OpsGenerator)


@pytest.fixture
def real_workpiece():
    wp = WorkPiece(Path("wp1"), b'<svg width="10" height="10" />', SvgImporter)
    wp.pos = 10, 20
    wp.set_size(40, 30)
    wp.angle = 90
    return wp


@pytest.mark.asyncio
async def test_generate_job_ops_assembles_correctly(
    doc, machine, mock_ops_generator, real_workpiece
):
    """
    Test that generate_job_ops correctly applies the workpiece's world
    transform matrix and then converts to machine coordinates.
    """
    # Arrange
    layer = doc.layers[0]
    with patch("rayforge.pipeline.steps.config", MagicMock()):
        step = create_outline_step(layer.workflow)

    # The new way to specify passes is via a post-assembly transformer
    multi_pass_transformer = MultiPassTransformer(passes=2)
    step.post_step_transformers_dicts = [multi_pass_transformer.to_dict()]

    layer.workflow.add_step(step)
    layer.add_workpiece(real_workpiece)

    base_ops = Ops()
    base_ops.move_to(0, 0)
    base_ops.line_to(10, 0)
    mock_ops_generator.get_ops.return_value = base_ops

    # Act
    final_ops = await generate_job_ops(doc, machine, mock_ops_generator)

    # Assert
    mock_ops_generator.get_ops.assert_called_once_with(step, real_workpiece)
    assert len([c for c in final_ops.commands if c.is_cutting_command()]) == 2

    # Verify transformations on the first pass
    # Workpiece: pos=(10,20), size=(40,30), angle=90.
    # The get_world_transform() method now uses -angle to preserve the
    # visual clockwise rotation convention. So, it applies a -90deg rotation.
    # Initial Ops points: (0,0) -> (10,0)
    #
    # 1. World Transform (Y-up, canonical space):
    #    The world matrix rotates -90deg (CW) around the center (20,15) and
    #    translates by (10,20).
    #    Point (0,0) -> (15, 55)
    #    Point (10,0) -> (15, 45)
    #
    # 2. Machine Transform (Y-down, height=150):
    #    (15, 55) -> scale(1,-1) -> (15, -55) -> translate(0,150) -> (15, 95)
    #    (15, 45) -> scale(1,-1) -> (15, -45) -> translate(0,150) -> (15, 105)
    move_cmds = [c for c in final_ops.commands if isinstance(c, MoveToCommand)]
    line_cmds = [c for c in final_ops.commands if isinstance(c, LineToCommand)]

    assert move_cmds[0].end == pytest.approx((15, 95, 0.0))
    assert line_cmds[0].end == pytest.approx((15, 105, 0.0))

    # Verify the second pass is identical
    assert move_cmds[1].end == pytest.approx((15, 95, 0.0))
    assert line_cmds[1].end == pytest.approx((15, 105, 0.0))


@pytest.mark.asyncio
async def test_job_generation_cancellation(doc, machine, mock_ops_generator):
    """Test that job generation can be cancelled via the context."""
    # Arrange
    layer = doc.layers[0]
    with patch("rayforge.pipeline.steps.config", MagicMock()):
        step = create_outline_step(layer.workflow)
    layer.workflow.add_step(step)

    wp1 = WorkPiece(Path("wp1"), b"", SvgImporter)
    wp1.set_size(10, 10)
    wp1.pos = 0, 0

    wp2 = WorkPiece(Path("wp2"), b"", SvgImporter)
    wp2.set_size(10, 10)
    wp2.pos = 20, 20

    # There are two workpieces, so total_items = 2
    layer.add_workpiece(wp1)
    layer.add_workpiece(wp2)

    mock_context = MagicMock()
    # is_cancelled() is checked at the start of the loop for each item.
    # To cancel before item 2, the 2nd call must return True.
    mock_context.is_cancelled.side_effect = [False, True]

    # Act & Assert
    with pytest.raises(CancelledError):
        await generate_job_ops(
            doc, machine, mock_ops_generator, context=mock_context
        )

    # After the exception is caught, verify what happened before the
    # cancellation.
    mock_ops_generator.get_ops.assert_called_once_with(step, wp1)

    # Verify progress was set for the first item (processed_items = 1).
    # Progress is 1 / total_items (2) = 0.5
    mock_context.set_progress.assert_called_once_with(0.5)
