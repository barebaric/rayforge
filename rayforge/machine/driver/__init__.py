import inspect
from typing import cast

from .driver import (
    DRIVER_MATURITY_LABELS,
    Driver,
    DriverMaturity,
    PWMParams,
)
from .dummy import NoDeviceDriver
from .grbl import (
    GrblNetworkDriver,
    GrblSerialDriver,
    GrblSerialSimpleDriver,
    GrblTelnetDriver,
)
from .marlin import MarlinSerialDriver
from .octoprint import OctoPrintDriver
from .ruida import RuidaDriver
from .smoothie import SmoothieDriver


def isdriver(obj):
    return (
        inspect.isclass(obj) and issubclass(obj, Driver) and obj is not Driver
    )


drivers = [
    cast(type[Driver], obj) for obj in list(locals().values()) if isdriver(obj)
]

driver_by_classname = {o.__name__: o for o in drivers}


def get_driver_cls(classname: str, default=NoDeviceDriver):
    return driver_by_classname.get(classname, default)


def register_driver(driver: type[Driver]):
    driver_by_classname[driver.__name__] = driver
    drivers.append(driver)


__all__ = [
    "DRIVER_MATURITY_LABELS",
    "Driver",
    "DriverMaturity",
    "GrblNetworkDriver",
    "GrblSerialDriver",
    "GrblSerialSimpleDriver",
    "GrblTelnetDriver",
    "MarlinSerialDriver",
    "NoDeviceDriver",
    "OctoPrintDriver",
    "PWMParams",
    "RuidaDriver",
    "SmoothieDriver",
]
