"""Shared test doubles for transformer tests."""


class FakeMachine:
    """Stand-in for Machine in to_spec tests.

    Attributes mirror what transformers query: kinematics, the
    driver's overscan flag, and the probed bed mesh.
    """

    def __init__(
        self,
        max_cut_speed: float = 6000.0,
        max_travel_speed: float = 12000.0,
        acceleration: float = 500.0,
        native_overscan: bool = False,
        bed_mesh=None,
    ):
        self.max_cut_speed = max_cut_speed
        self.max_travel_speed = max_travel_speed
        self.acceleration = acceleration
        self.driver = type(
            "Driver", (), {"native_overscan": native_overscan}
        )()
        self.bed_mesh = bed_mesh
