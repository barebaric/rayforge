import inspect
from typing import Type, cast
from .driver import Driver
from .dummy import NoDeviceDriver
from .grbl import GrblDriver
from .grbl_serial import GrblSerialDriver
from .grbl_next import GrblNextNetworkDriver
from .grbl_serial_next import GrblNextSerialDriver
from .smoothie import SmoothieDriver


def isdriver(obj):
    return (
        inspect.isclass(obj) and issubclass(obj, Driver) and obj is not Driver
    )


drivers = [
    cast(Type[Driver], obj) for obj in list(locals().values()) if isdriver(obj)
]

driver_by_classname = dict([(o.__name__, o) for o in drivers])


def get_driver_cls(classname: str, default=NoDeviceDriver):
    return driver_by_classname.get(classname, default)


def get_driver(classname: str, default=NoDeviceDriver):
    return get_driver_cls(classname, default)()


__all__ = [
    "Driver",
    "NoDeviceDriver",
    "GrblDriver",
    "GrblSerialDriver",
    "GrblNextNetworkDriver",
    "GrblNextSerialDriver",
    "SmoothieDriver",
]
