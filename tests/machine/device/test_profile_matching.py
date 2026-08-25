from pathlib import Path

from rayforge.machine.device.manager import DeviceProfileManager
from rayforge.machine.driver.discovery import (
    GENERIC_TOKENS,
    DeviceIdentity,
    normalize_tokens,
)

_PROFILE_YAML = """\
api_version: 1
device:
  name: {name}
  vendor: {vendor}
machine:
  driver: RuidaDriver
"""


def _make_profile(root: Path, dirname: str, name: str, vendor: str) -> Path:
    profile_dir = root / dirname
    profile_dir.mkdir(parents=True)
    (profile_dir / "device.yaml").write_text(
        _PROFILE_YAML.format(name=name, vendor=vendor)
    )
    return profile_dir


def _manager(tmp_path: Path) -> DeviceProfileManager:
    mgr = DeviceProfileManager(source_dirs=[tmp_path])
    mgr.discover()
    return mgr


def _identity_for(*texts: str) -> DeviceIdentity:
    return DeviceIdentity(tokens=normalize_tokens(*texts))


def test_unique_vendor_match(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    _make_profile(tmp_path, "other-two", "Other Two", "Gadgetco")
    mgr = _manager(tmp_path)

    identity = _identity_for("Frobnicate Laser Master", "CH340")
    matched = mgr.match_device(identity)
    assert matched is not None
    assert matched.name == "Frob One"


def test_ambiguous_same_vendor_returns_none(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    _make_profile(tmp_path, "frob-two", "Frob Two", "Frobnicate")
    mgr = _manager(tmp_path)

    identity = _identity_for("Frobnicate laser")
    assert mgr.match_device(identity) is None


def test_model_tokens_narrow_the_match(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    frob_pro = tmp_path / "frob-pro"
    frob_pro.mkdir()
    (frob_pro / "device.yaml").write_text(
        _PROFILE_YAML.format(name="Frob Pro", vendor="Frobnicate").replace(
            "machine:", "  model: Pro\nmachine:"
        )
    )
    mgr = _manager(tmp_path)

    # Model tokens present in the identity: the model-less profile
    # matches on vendor alone, the "Pro" profile does not (its model
    # token is missing), so the match is unambiguous.
    identity = _identity_for("Frobnicate laser")
    matched = mgr.match_device(identity)
    assert matched is not None
    assert matched.name == "Frob One"

    # With the model token, the more specific "Pro" profile outranks
    # the model-less one.
    identity_pro = _identity_for("Frobnicate Pro laser")
    matched_pro = mgr.match_device(identity_pro)
    assert matched_pro is not None
    assert matched_pro.name == "Frob Pro"


def test_no_match_returns_none(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    mgr = _manager(tmp_path)

    assert mgr.match_device(_identity_for("USB Serial CH340")) is None
    assert mgr.match_device(DeviceIdentity()) is None


def test_generic_vendor_is_ignored(tmp_path):
    _make_profile(tmp_path, "generic", "Generic USB Serial", "USB Serial")
    mgr = _manager(tmp_path)

    identity = _identity_for("USB Serial CH340")
    assert mgr.match_device(identity) is None
    assert normalize_tokens("USB Serial") <= GENERIC_TOKENS


def test_builtin_profiles_all_declare_vendor():
    import rayforge

    devices_dir = Path(rayforge.__file__).parent / "resources" / "devices"
    assert devices_dir.is_dir()
    mgr = DeviceProfileManager(source_dirs=[devices_dir])
    mgr.discover()

    profiles = mgr.get_all()
    assert profiles, "expected built-in device profiles"
    missing = [p.name for p in profiles if not (p.meta.vendor or "").strip()]
    assert missing == []


def test_machine_name_matches_builtin_profile():
    """A device name announced by the firmware itself (Grbl's
    [MSG:machine:...] line) drives matching against the built-in
    profiles."""
    import rayforge
    from rayforge.machine.driver.discovery import build_identity

    devices_dir = Path(rayforge.__file__).parent / "resources" / "devices"
    mgr = DeviceProfileManager(source_dirs=[devices_dir])
    mgr.discover()

    identity = build_identity(
        "grbl",
        b"[VER:1.0.15,20240923:]\r\n[MSG:mechine:Sculpfun iCube]\r\n",
        port_info=None,
        device_name="Sculpfun iCube",
    )
    matched = mgr.match_device(identity)
    assert matched is not None
    assert matched.name == "Sculpfun iCube"
