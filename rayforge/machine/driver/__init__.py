import inspect
from typing import cast

from raygeo.driver.core.discovery import (
    DeviceIdentity,
    DiscoveredDevice,
)
from raygeo.driver.core.discovery.engine import find_all_devices
from raygeo.driver.core.discovery.mdns_channel import find_network_devices
from raygeo.driver.core.discovery.serial_channel import normalize_tokens

from .driver import (
    DRIVER_MATURITY_LABELS,
    Driver,
    DriverMaturity,
    PWMParams,
)
from .dummy import NoDeviceDriver
from .grbl import (
    GrblBluetoothDriver,
    GrblNetworkDriver,
    GrblSerialDriver,
    GrblSerialSimpleDriver,
    GrblTelnetDriver,
)
from .marlin import MarlinSerialDriver
from .octoprint import OctoPrintDriver
from .remote_driver import RaydriverRemoteDriver
from .ruidarpa import RuidaRPAAdapter
from .smoothie import SmoothieDriver
from .xtool import XtoolM2Driver, XtoolNetworkDriver


def isdriver(obj):
    return (
        inspect.isclass(obj) and issubclass(obj, Driver) and obj is not Driver
    )


drivers = [
    cast(type[Driver], obj) for obj in list(locals().values()) if isdriver(obj)
]

driver_by_classname = {o.__name__: o for o in drivers}

# Transitional alias, remove with the next release: machines saved
# while the Rust-backed serial driver carried its experimental name
# keep resolving.
driver_by_classname["GrblSerialNextDriver"] = GrblSerialDriver


def get_driver_cls(classname: str, default=NoDeviceDriver):
    return driver_by_classname.get(classname, default)


def find_driver_cls(classname: str) -> type[Driver] | None:
    """Return the registered driver class for *classname*, or ``None``.

    Unlike :func:`get_driver_cls` this does not substitute the dummy
    driver, so callers can tell a configured-but-missing driver from a
    deliberately driverless machine.
    """
    return driver_by_classname.get(classname)


def register_driver(driver: type[Driver]):
    driver_by_classname[driver.__name__] = driver
    drivers.append(driver)


__all__ = [
    "DRIVER_MATURITY_LABELS",
    "DeviceIdentity",
    "DiscoveredDevice",
    "Driver",
    "DriverMaturity",
    "GrblBluetoothDriver",
    "GrblNetworkDriver",
    "GrblSerialDriver",
    "GrblSerialSimpleDriver",
    "GrblTelnetDriver",
    "MarlinSerialDriver",
    "NoDeviceDriver",
    "OctoPrintDriver",
    "PWMParams",
    "RaydriverRemoteDriver",
    "RuidaRPAAdapter",
    "SmoothieDriver",
    "XtoolM2Driver",
    "XtoolNetworkDriver",
    "find_all_devices",
    "find_network_devices",
    "normalize_tokens",
]
