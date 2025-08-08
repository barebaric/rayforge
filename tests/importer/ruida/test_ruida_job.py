import pytest
from rayforge.importer.ruida.job import RuidaJob, RuidaLayer, RuidaCommand


def test_ruidajob_creation():
    """Tests basic creation of a RuidaJob and its components."""
    job = RuidaJob()
    assert isinstance(job.layers, dict)
    assert isinstance(job.commands, list)
    assert len(job.layers) == 0
    assert len(job.commands) == 0


def test_ruidajob_population():
    """Tests adding layers and commands to a RuidaJob."""
    job = RuidaJob()

    # Define and add a layer
    layer0 = RuidaLayer(color_index=0, speed=50.0, power=75.5)
    job.layers[0] = layer0

    # Add commands
    job.commands.append(
        RuidaCommand(command_type="Move_Abs", params=[0.0, 0.0], color_index=0)
    )
    job.commands.append(
        RuidaCommand(command_type="Cut_Abs", params=[10.0, 0.0], color_index=0)
    )
    job.commands.append(
        RuidaCommand(
            command_type="Cut_Abs", params=[10.0, 10.0], color_index=0
        )
    )

    assert len(job.layers) == 1
    assert job.layers[0].speed == 50.0
    assert len(job.commands) == 3
    assert job.commands[1].command_type == "Cut_Abs"
    assert job.commands[2].params == [10.0, 10.0]


def test_ruidajob_get_extents():
    """Tests the bounding box calculation."""
    job = RuidaJob()
    job.commands.extend(
        [
            RuidaCommand(command_type="Move_Abs", params=[10.0, 20.0]),
            RuidaCommand(command_type="Cut_Abs", params=[100.0, 20.0]),
            RuidaCommand(command_type="Cut_Abs", params=[100.0, 50.0]),
            RuidaCommand(command_type="Cut_Abs", params=[-5.0, 50.0]),
        ]
    )

    min_x, min_y, max_x, max_y = job.get_extents()

    assert min_x == pytest.approx(-5.0)
    assert min_y == pytest.approx(20.0)
    assert max_x == pytest.approx(100.0)
    assert max_y == pytest.approx(50.0)


def test_ruidajob_get_extents_no_points():
    """Tests that extents are zero if there are no geometric commands."""
    job = RuidaJob()
    job.commands.append(RuidaCommand(command_type="End"))
    assert job.get_extents() == (0.0, 0.0, 0.0, 0.0)
