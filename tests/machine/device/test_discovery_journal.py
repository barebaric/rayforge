import json

from rayforge.machine.device import discovery_journal
from rayforge.machine.device.matching import ProfileMatch
from rayforge.machine.device.profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)
from rayforge.machine.discovery import DeviceIdentity


def _profile(name: str) -> DeviceProfile:
    return DeviceProfile(
        meta=DeviceMeta(name=name, vendor="Frobnicate"),
        machine_config=MachineConfig(),
        dialect_config={},
    )


def test_record_and_read_roundtrip(tmp_path):
    path = tmp_path / discovery_journal.FILENAME
    identity = DeviceIdentity(
        firmware="grbl",
        banner="Sculpfun iCube",
        usb_vid=0x1A86,
        usb_pid=0x7523,
    )
    discovery_journal.record_match(path, identity, [])

    entries = discovery_journal.read_entries(path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["firmware"] == "grbl"
    assert entry["banner"] == "Sculpfun iCube"
    assert entry["usb_vid"] == "1a86"
    assert entry["usb_pid"] == "7523"
    assert "timestamp" in entry


def test_matches_are_recorded(tmp_path):
    path = tmp_path / discovery_journal.FILENAME
    identity = DeviceIdentity(tokens=frozenset({"sculpfun"}))
    matches = [
        ProfileMatch(profile=_profile("Frob One"), confidence=0.8),
        ProfileMatch(profile=_profile("Frob Two"), confidence=0.6),
    ]
    discovery_journal.record_match(path, identity, matches)

    entry = discovery_journal.read_entries(path)[0]
    assert entry["tokens"] == ["sculpfun"]
    assert entry["matches"] == [
        {"profile": "Frob One", "confidence": 0.8},
        {"profile": "Frob Two", "confidence": 0.6},
    ]
    lines = path.read_text().splitlines()
    json.loads(lines[0])


def test_corrupted_lines_are_skipped(tmp_path):
    path = tmp_path / discovery_journal.FILENAME
    identity = DeviceIdentity()
    discovery_journal.record_match(path, identity, [])
    with open(path, "a") as f:
        f.write("not json\n")
    discovery_journal.record_match(path, identity, [])

    entries = discovery_journal.read_entries(path)
    assert len(entries) == 2


def test_journal_is_trimmed_to_max_entries(tmp_path):
    path = tmp_path / discovery_journal.FILENAME
    identity = DeviceIdentity()
    for _ in range(discovery_journal.MAX_ENTRIES + 50):
        discovery_journal.record_match(path, identity, [])

    entries = discovery_journal.read_entries(path)
    assert len(entries) == discovery_journal.MAX_ENTRIES


def test_unwritable_path_does_not_raise(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("file")
    path = blocker / discovery_journal.FILENAME
    discovery_journal.record_match(path, DeviceIdentity(), [])
