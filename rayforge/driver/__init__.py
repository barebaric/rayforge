# flake8: noqa:F401
import inspect
from .driver import Driver
from .dummy import NoDeviceDriver
from .grbl import GrblDriver
from .grbl_serial import GrblSerialDriver

def isdriver(obj):
    return (inspect.isclass(obj)
            and issubclass(obj, Driver)
            and obj is not Driver)

drivers = [obj for obj in list(locals().values())
           if isdriver(obj)]

driver_by_classname = dict([(o.__name__, o) for o in drivers])

def get_driver_cls(classname: str, default=NoDeviceDriver):
    return driver_by_classname.get(classname, default)

def get_driver(classname: str, default=NoDeviceDriver):
    return get_driver_cls(classname, default)()

def get_params(driver_cls):
    signature = inspect.signature(driver_cls.setup)
    return signature.parameters.items()

__all__ = [
    'Driver',
    'NoDeviceDriver',
    'GrblDriver',
    'GrblSerialDriver',
]
