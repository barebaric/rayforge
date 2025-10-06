import unittest
from rayforge.core.ops import Ops
from rayforge.pipeline.artifact import Artifact
from rayforge.pipeline import CoordinateSystem


class TestArtifactBase(unittest.TestCase):
    """Test suite for the abstract Artifact base class."""

    def test_cannot_instantiate_abc(self):
        """Ensures the abstract base class cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            # We are intentionally trying to instantiate an abstract class to
            # verify that it raises a TypeError. Pylance correctly flags this
            # as an error statically, so we ignore it for this test case.
            Artifact(  # type: ignore
                ops=Ops(),
                is_scalable=True,
                source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            )

    def test_from_dict_is_abstract(self):
        """Ensures the from_dict classmethod is abstract."""
        with self.assertRaises(NotImplementedError):
            # We are testing that calling the abstract method raises an error.
            # Pylance flags this, so we ignore it for this specific test.
            Artifact.from_dict({})  # type: ignore

    def test_to_dict_on_subclass(self):
        """Tests that the base to_dict method works correctly on a subclass."""

        # Create a minimal concrete subclass for testing purposes
        class DummyArtifact(Artifact):
            @classmethod
            def from_dict(cls, data):
                # This implementation satisfies the abstract method's type
                # hint, even though it's not used in this specific test.
                return cls(
                    ops=Ops(),
                    is_scalable=False,
                    source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
                )

        ops = Ops()
        ops.move_to(1, 2, 3)
        artifact = DummyArtifact(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            source_dimensions=(100, 200),
            generation_size=(50, 100),
        )
        artifact.type = "dummy"

        expected_dict = {
            "type": "dummy",
            "ops": ops.to_dict(),
            "is_scalable": False,
            "source_coordinate_system": "PIXEL_SPACE",
            "source_dimensions": (100, 200),
            "generation_size": (50, 100),
        }

        self.assertDictEqual(artifact.to_dict(), expected_dict)


if __name__ == "__main__":
    unittest.main()
