import pytest

from rayforge.machine.discovery import (
    DeviceIdentity,
    DiscoveredDevice,
)


def test_discovered_device_key():
    device = DiscoveredDevice(
        driver_name="GrblSerialDriver",
        params={"port": "/dev/ttyUSB0", "baudrate": 115200},
        label="GRBL device",
        detail="/dev/ttyUSB0 at 115200 baud",
    )
    assert device.key == "GrblSerialDriver:/dev/ttyUSB0"
    assert DeviceIdentity() == DeviceIdentity()


def test_discovered_device_key_network_unique_per_host():
    def _device(host):
        return DiscoveredDevice(
            driver_name="OctoPrintDriver",
            params={"host": host, "port": 80},
            label="OctoPrint",
            detail=f"{host}:80",
        )

    # Two servers behind the same TCP port stay distinguishable.
    assert _device("192.168.1.42").key != _device("192.168.1.43").key
    assert _device("192.168.1.42").key == "OctoPrintDriver:192.168.1.42:80"


@pytest.mark.parametrize(
    "params,expected",
    [
        ({"host": "h", "port": 80}, "D:h:80"),
        ({}, "D:None"),
        ({"port": 1234}, "D:1234"),
    ],
)
def test_discovered_device_key_fallbacks(params, expected):
    device = DiscoveredDevice(
        driver_name="D", params=params, label="", detail=""
    )
    assert device.key == expected
