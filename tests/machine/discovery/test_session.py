from rayforge.machine.discovery import (
    DeviceIdentity,
    DiscoveredDevice,
    DiscoverySession,
)


def _serial_device(port, key_driver="GrblSerialDriver"):
    return DiscoveredDevice(
        driver_name=key_driver,
        params={"port": port, "baudrate": 115200},
        label="GRBL device",
        detail=f"{port} at 115200 baud",
    )


def _network_device(host, port=80, driver="OctoPrintDriver"):
    return DiscoveredDevice(
        driver_name=driver,
        params={"host": host, "port": port},
        label="OctoPrint",
        detail=f"{host}:{port}",
    )


def test_apply_scan_reports_add_and_remove():
    session = DiscoverySession()
    added, removed = session.apply_scan([_serial_device("/dev/ttyUSB0")])
    assert [d.key for d in added] == ["GrblSerialDriver:/dev/ttyUSB0"]
    assert removed == []

    added, removed = session.apply_scan([_network_device("192.168.1.42")])
    assert [d.key for d in added] == ["OctoPrintDriver:192.168.1.42:80"]
    assert removed == []

    # Network devices are not held, so disappearing from a rescan
    # drops them immediately.
    added, removed = session.apply_scan([_network_device("192.168.1.43")])
    assert [d.key for d in added] == ["OctoPrintDriver:192.168.1.43:80"]
    assert removed == ["OctoPrintDriver:192.168.1.42:80"]


def test_apply_scan_deduplicates_within_one_result():
    session = DiscoverySession()
    device = _serial_device("/dev/ttyUSB0")
    added, _ = session.apply_scan([device, device])
    assert len(added) == 1


def test_held_ports_survive_absence_from_rescan():
    """Devices on held ports are excluded from rescans; their absence
    from a result must not drop them."""
    session = DiscoverySession()
    session.apply_scan([_serial_device("/dev/ttyUSB0")])
    assert session.held_ports == {"/dev/ttyUSB0"}

    added, removed = session.apply_scan([])
    assert added == []
    assert removed == []
    assert len(session.devices) == 1


def test_network_device_with_string_port_is_never_held_or_pruned():
    """A device whose params carry both a ``host`` and a *string*
    port is still a network device: its port must never enter the
    held set (so absence from a rescan drops it right away) and the
    serial-port pruning list can never remove it."""
    session = DiscoverySession()
    device = DiscoveredDevice(
        driver_name="GrblNetworkDriver",
        params={"host": "grbl.local", "port": "80"},
        label="GRBL network",
        detail="grbl.local:80",
    )

    added, _ = session.apply_scan([device])
    assert [d.key for d in added] == [device.key]
    assert session.held_ports == set()

    # Absence from a rescan drops it immediately (unlike serial).
    added, removed = session.apply_scan([])
    assert added == []
    assert removed == [device.key]

    # Re-adding and pruning with an empty serial list leaves it.
    session.apply_scan([device])
    assert session.prune_absent_ports(set()) == []
    assert len(session.devices) == 1


def test_prune_absent_ports_drops_only_serial_devices():
    session = DiscoverySession()
    session.apply_scan(
        [
            _serial_device("/dev/ttyUSB0"),
            _network_device("192.168.1.42"),
        ]
    )
    # Network devices have no serial port and are never pruned.
    removed = session.prune_absent_ports(set())
    assert removed == ["GrblSerialDriver:/dev/ttyUSB0"]
    assert session.held_ports == set()
    assert [d.params.get("host") for d in session.devices] == ["192.168.1.42"]


def test_prune_keeps_present_ports():
    session = DiscoverySession()
    session.apply_scan([_serial_device("/dev/ttyUSB0")])
    removed = session.prune_absent_ports({"/dev/ttyUSB0", "/dev/x"})
    assert removed == []
    assert session.held_ports == {"/dev/ttyUSB0"}


def test_apply_probe_enriches_device():
    session = DiscoverySession()
    session.apply_scan([_serial_device("/dev/ttyUSB0")])

    class FakeProfile:
        pass

    profile = FakeProfile()
    enriched = session.apply_probe("GrblSerialDriver:/dev/ttyUSB0", profile)
    assert enriched is not None
    assert enriched.probe_profile is profile
    assert session.get("GrblSerialDriver:/dev/ttyUSB0") is enriched

    # Unknown keys are ignored.
    assert session.apply_probe("nope", profile) is None


def test_reset_clears_everything():
    session = DiscoverySession()
    session.apply_scan([_serial_device("/dev/ttyUSB0"), _network_device("h1")])
    session.reset()
    assert session.devices == []
    assert session.held_ports == set()


def test_identity_defaults_unchanged():
    assert DeviceIdentity() == DeviceIdentity()
