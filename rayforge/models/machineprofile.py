from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from .machine import Machine, Laser


@dataclass
class MachineProfile:
    """
    A template for creating a new Machine instance with pre-configured
    settings. All fields are optional to allow for partial profiles.
    """

    name: Optional[str] = None
    driver_class_name: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None
    y_axis_down: Optional[bool] = None
    max_travel_speed: Optional[int] = None
    max_cut_speed: Optional[int] = None
    preamble: Optional[List[str]] = None
    postscript: Optional[List[str]] = None
    air_assist_on: Optional[str] = None
    air_assist_off: Optional[str] = None
    driver_args: Optional[Dict[str, Any]] = None

    def create_machine(self) -> Machine:
        """
        Creates a Machine instance from this profile.

        Only attributes that are not None in the profile will be applied
        to the new Machine instance, allowing the Machine's own defaults to
        be used for any unspecified profile values.
        """
        m = Machine()

        if self.name is not None:
            m.name = self.name
        if self.driver_class_name is not None:
            m.driver = self.driver_class_name
        if self.dimensions is not None:
            m.dimensions = self.dimensions
        if self.y_axis_down is not None:
            m.y_axis_down = self.y_axis_down
        if self.max_travel_speed is not None:
            m.max_travel_speed = self.max_travel_speed
        if self.max_cut_speed is not None:
            m.max_cut_speed = self.max_cut_speed
        if self.preamble is not None:
            m.preamble = self.preamble.copy()
        if self.postscript is not None:
            m.postscript = self.postscript.copy()
        if self.air_assist_on is not None:
            m.air_assist_on = self.air_assist_on
        if self.air_assist_off is not None:
            m.air_assist_off = self.air_assist_off
        if self.driver_args is not None:
            m.driver_args = self.driver_args.copy()

        # Standard setup for any machine created from a profile
        m.heads = []
        m.add_head(Laser())
        m.cameras = []
        return m


PROFILES: List[MachineProfile] = [
    MachineProfile(
        name="Sculpfun iCube",
        driver_class_name="GrblDriver",
        dimensions=(120, 120),
        y_axis_down=False,
        max_travel_speed=3000,
        max_cut_speed=1000,
        preamble=["G21", "G90"],
        postscript=["G0 X0 Y0"],
        air_assist_on="M8",
        air_assist_off="M9",
    ),
    MachineProfile(
        name="Custom GRBL Machine",
        driver_class_name="GrblDriver",
        y_axis_down=False,
        preamble=["G21", "G90", "$32=1"],
        postscript=["$32=0", "G0 X0 Y0"],
    ),
]
