import pytest
from unittest.mock import MagicMock, patch
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.ops import Ops, LineToCommand, MoveToCommand
from rayforge.machine.models.machine import Machine, Laser
from rayforge.pipeline.generator import OpsGenerator
from rayforge.shared.tasker.manager import CancelledError
from rayforge.pipeline.job import generate_job_ops
from rayforge.pipeline.steps import create_outline_step
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
    d.layers[0].workflow.set_steps([])
    return d


@pytest.fixture
def mock_ops_generator():
    """A mock OpsGenerator that we can configure for tests."""
    return MagicMock(spec=OpsGenerator)


@pytest.fixture
def real_workpiece():
    wp = WorkPiece("wp1", b'<svg width="10" height="10" />', SvgImporter)
    wp.pos = (10, 20)
    wp.size = (40, 30)
    wp.angle = 90
    return wp


@pytest.mark.asyncio
async def test_generate_job_ops_assembles_correctly(
    doc, machine, mock_ops_generator, real_workpiece
):
    """
    Test that generate_job_ops correctly rotates, translates, and
    transforms Ops from the generator into the final job.
    """
    # Arrange
    layer = doc.layers[0]
    with patch("rayforge.pipeline.steps.config", MagicMock()):
        step = create_outline_step(layer.workflow)
    step.passes = 2
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
    # Initial: (0,0) -> (10,0)
    # Rotated -90deg around (20,15): (5,35) -> (5,25)
    # Translated by (10,20): (15,55) -> (15,45)
    # Y-flipped for machine: (15, 95) -> (15, 105)
    move_cmds = [c for c in final_ops.commands if isinstance(c, MoveToCommand)]
    line_cmds = [c for c in final_ops.commands if isinstance(c, LineToCommand)]

    assert move_cmds[0].end == pytest.approx((15, 95))
    assert line_cmds[0].end == pytest.approx((15, 105))

    # Verify the second pass is identical
    assert move_cmds[1].end == pytest.approx((15, 95))
    assert line_cmds[1].end == pytest.approx((15, 105))


@pytest.mark.asyncio
async def test_job_generation_cancellation(doc, machine, mock_ops_generator):
    """Test that job generation can be cancelled via the context."""
    # Arrange
    layer = doc.layers[0]
    with patch("rayforge.pipeline.steps.config", MagicMock()):
        step = create_outline_step(layer.workflow)
    layer.workflow.add_step(step)

    wp1 = WorkPiece("wp1", b"", SvgImporter)
    wp1.size = (10, 10)
    wp1.pos = (0, 0)  # FIX: Add position to make workpiece "renderable"

    wp2 = WorkPiece("wp2", b"", SvgImporter)
    wp2.size = (10, 10)
    wp2.pos = (20, 20)  # FIX: Add position to make workpiece "renderable"

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
    mock_context.set_progress.assert_called_once_with(0.0 / 2.0)
