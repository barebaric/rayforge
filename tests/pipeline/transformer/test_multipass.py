import pytest
from typing import List
from rayforge.core.ops import Ops, Command, LineToCommand, MoveToCommand
from rayforge.pipeline.transformer.multipass import MultiPassTransformer


class TestMultiPassTransformer:
    """
    Tests the functionality of the MultiPassTransformer, which repeats
    an Ops object's commands to create multiple passes.
    """

    def test_duplicates_commands_without_z_step(self):
        """
        Tests that commands are duplicated the correct number of times when
        no z_step_down is applied.
        """
        # Arrange
        # Explicitly type the list to satisfy the type checker
        # (List is invariant)
        initial_commands: List[Command] = [
            MoveToCommand(end=(10, 10, 0)),
            LineToCommand(end=(20, 20, 0)),
        ]
        ops = Ops()
        ops.commands = initial_commands

        transformer = MultiPassTransformer(passes=3, z_step_down=0.0)

        # Act
        transformer.run(ops)

        # Assert
        # Original commands + 2 copies = 3 total passes
        assert len(ops.commands) == 6

        # Verify the sequence and types of commands
        assert isinstance(ops.commands[0], MoveToCommand)
        assert isinstance(ops.commands[1], LineToCommand)
        assert isinstance(ops.commands[2], MoveToCommand)
        assert isinstance(ops.commands[3], LineToCommand)
        assert isinstance(ops.commands[4], MoveToCommand)
        assert isinstance(ops.commands[5], LineToCommand)

        # The first pass has original positions
        assert ops.commands[0].end == (10, 10, 0)
        assert ops.commands[1].end == (20, 20, 0)
        # All subsequent passes also have original positions (no z-step)
        assert ops.commands[2].end == (10, 10, 0)
        assert ops.commands[3].end == (20, 20, 0)
        assert ops.commands[4].end == (10, 10, 0)
        assert ops.commands[5].end == (20, 20, 0)

    def test_applies_z_step_down_for_each_pass(self):
        """
        Tests that z_step_down correctly modifies the Z coordinate on
        each subsequent pass.
        """
        # Arrange
        initial_commands: List[Command] = [LineToCommand(end=(10, 10, 5.0))]
        ops = Ops()
        ops.commands = initial_commands
        num_passes = 3
        z_step = 0.5

        transformer = MultiPassTransformer(
            passes=num_passes, z_step_down=z_step
        )

        # Act
        transformer.run(ops)

        # Assert
        assert len(ops.commands) == 3

        # Add assertions to assure the type checker that .end is not None
        assert ops.commands[0].end is not None
        assert ops.commands[1].end is not None
        assert ops.commands[2].end is not None

        # Pass 1 (original): Z should be untouched
        assert ops.commands[0].end[2] == 5.0
        # Pass 2: Z should be original_z - (1 * z_step)
        assert ops.commands[1].end[2] == pytest.approx(5.0 - 0.5)
        # Pass 3: Z should be original_z - (2 * z_step)
        assert ops.commands[2].end[2] == pytest.approx(5.0 - 1.0)

    def test_no_op_for_single_pass_and_no_z_step(self):
        """
        Tests the optimization that if passes=1 and z_step_down=0, the
        run method does nothing.
        """
        # Arrange
        initial_commands: List[Command] = [MoveToCommand(end=(0, 0, 0))]
        ops = Ops()
        ops.commands = initial_commands
        # Keep a reference to the original list object
        original_commands_list = ops.commands

        transformer = MultiPassTransformer(passes=1, z_step_down=0.0)

        # Act
        transformer.run(ops)

        # Assert
        # The list object itself should not have been replaced.
        assert ops.commands is original_commands_list
        assert len(ops.commands) == 1

    def test_no_op_for_empty_commands(self):
        """
        Tests that if the initial Ops object has no commands, the transformer
        does nothing.
        """
        # Arrange
        ops = Ops()
        ops.commands = []
        transformer = MultiPassTransformer(passes=5)

        # Act
        transformer.run(ops)

        # Assert
        assert len(ops.commands) == 0

    def test_passes_property_validation(self):
        """
        Tests that the 'passes' property setter enforces a minimum value of 1.
        """
        # Arrange
        transformer = MultiPassTransformer()

        # Act & Assert
        transformer.passes = 5
        assert transformer.passes == 5

        transformer.passes = 0
        assert transformer.passes == 1

        transformer.passes = -10
        assert transformer.passes == 1

        transformer.passes = 1
        assert transformer.passes == 1

    def test_serialization_and_deserialization(self):
        """
        Tests that the transformer can be serialized to a dict and
        recreated from that dict.
        """
        # Arrange
        transformer = MultiPassTransformer(
            enabled=False, passes=4, z_step_down=1.23
        )

        # Act
        data = transformer.to_dict()
        recreated_transformer = MultiPassTransformer.from_dict(data)

        # Assert
        assert data["name"] == "MultiPassTransformer"
        assert data["enabled"] is False
        assert data["passes"] == 4
        assert data["z_step_down"] == 1.23

        assert isinstance(recreated_transformer, MultiPassTransformer)
        assert recreated_transformer.enabled is False
        assert recreated_transformer.passes == 4
        assert recreated_transformer.z_step_down == 1.23
