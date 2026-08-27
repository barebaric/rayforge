from pathlib import Path

import pytest
import yaml

import rayforge
from rayforge.machine.device import discovery_journal
from rayforge.machine.device.manager import DeviceProfileManager
from rayforge.machine.device.matching import (
    CONFIDENCE_CERTAIN,
    CONFIDENCE_GENERIC_USB_ID,
    CONFIDENCE_MODEL_TOKENS,
    CONFIDENCE_VENDOR_TOKENS,
    CONFIDENCE_VID_ONLY,
    ProfileMatch,
    certain_match,
)
from rayforge.machine.device.profile import DeviceProfile, parse_meta
from rayforge.machine.discovery import (
    GENERIC_TOKENS,
    DeviceIdentity,
    build_identity,
    normalize_tokens,
)

#: A fictional vendor-specific vid/pid pair (not a stock USB-serial
#: bridge chip).
CUSTOM_VID = 0xABCD
CUSTOM_PID = 0x1234

_PROFILE_YAML = """\
api_version: 1
device:
  name: {name}
  vendor: {vendor}{extra}
machine:
  driver: RuidaDriver
"""


def _make_profile(
    root: Path,
    dirname: str,
    name: str,
    vendor: str,
    model: str | None = None,
    usb_ids: list[str] | None = None,
) -> Path:
    extra = ""
    if model is not None:
        extra += f"\n  model: {model}"
    if usb_ids is not None:
        ids = "".join(f"\n    - {uid}" for uid in usb_ids)
        extra += f"\n  usb_ids:{ids}"
    profile_dir = root / dirname
    profile_dir.mkdir(parents=True)
    (profile_dir / "device.yaml").write_text(
        _PROFILE_YAML.format(name=name, vendor=vendor, extra=extra)
    )
    return profile_dir


def _manager(tmp_path: Path) -> DeviceProfileManager:
    mgr = DeviceProfileManager(source_dirs=[tmp_path])
    mgr.discover()
    return mgr


def _identity_for(*texts: str) -> DeviceIdentity:
    return DeviceIdentity(tokens=normalize_tokens(*texts))


def _usb_identity(
    vid: int | None = None, pid: int | None = None, *texts: str
) -> DeviceIdentity:
    return DeviceIdentity(
        tokens=normalize_tokens(*texts), usb_vid=vid, usb_pid=pid
    )


def _matches(mgr, identity) -> list[ProfileMatch]:
    return mgr.match_device(identity)


def _names(matches: list[ProfileMatch]) -> list[str]:
    return [m.profile.name for m in matches]


def test_unique_vendor_match(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    _make_profile(tmp_path, "other-two", "Other Two", "Gadgetco")
    mgr = _manager(tmp_path)

    identity = _identity_for("Frobnicate Laser Master", "CH340")
    matches = _matches(mgr, identity)
    assert _names(matches) == ["Frob One"]
    assert matches[0].confidence == CONFIDENCE_VENDOR_TOKENS


def test_ambiguous_same_vendor_lists_both(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    _make_profile(tmp_path, "frob-two", "Frob Two", "Frobnicate")
    mgr = _manager(tmp_path)

    identity = _identity_for("Frobnicate laser")
    matches = _matches(mgr, identity)
    assert matches[0].confidence == CONFIDENCE_VENDOR_TOKENS
    assert set(_names(matches)) == {"Frob One", "Frob Two"}
    assert certain_match(matches) is None


def test_model_tokens_rank_higher(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    _make_profile(tmp_path, "frob-pro", "Frob Pro", "Frobnicate", model="Pro")
    mgr = _manager(tmp_path)

    # Model tokens present in the identity: both profiles match, but
    # the model-specific one outranks the model-less one.
    identity_pro = _identity_for("Frobnicate Pro laser")
    matches = _matches(mgr, identity_pro)
    assert _names(matches) == ["Frob Pro", "Frob One"]
    assert matches[0].confidence == CONFIDENCE_MODEL_TOKENS

    # With the full product name the device effectively announces
    # itself, which is a certain match. The model-less profile stays
    # a ranked candidate below it.
    identity_named = _identity_for("Frobnicate Frob Pro laser")
    matches = _matches(mgr, identity_named)
    assert _names(matches) == ["Frob Pro", "Frob One"]
    assert matches[0].confidence == CONFIDENCE_CERTAIN
    assert certain_match(matches) is not None

    # Without the model token both profiles fall back to their
    # vendor evidence: a declared model missing from the identity is
    # not negative evidence.
    identity = _identity_for("Frobnicate laser")
    matches = _matches(mgr, identity)
    assert _names(matches) == ["Frob One", "Frob Pro"]
    assert all(m.confidence == CONFIDENCE_VENDOR_TOKENS for m in matches)


def test_name_with_dropped_tokens_is_not_certain(tmp_path):
    """A profile whose display name loses its distinctive word to
    tokenization ("Sculpfun C1" keeps only "sculpfun") must not earn
    certainty from the vendor token alone."""
    _make_profile(tmp_path, "c-one", "Frobnicate c1", "Frobnicate")
    mgr = _manager(tmp_path)

    identity = _identity_for("frobnicate laser")
    matches = _matches(mgr, identity)
    assert matches[0].confidence == CONFIDENCE_VENDOR_TOKENS
    assert certain_match(matches) is None


def test_no_match_returns_empty(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate")
    mgr = _manager(tmp_path)

    assert mgr.match_device(_identity_for("USB Serial CH340")) == []
    assert mgr.match_device(DeviceIdentity()) == []


def test_generic_vendor_is_ignored(tmp_path):
    _make_profile(tmp_path, "generic", "Generic USB Serial", "USB Serial")
    mgr = _manager(tmp_path)

    identity = _identity_for("USB Serial CH340")
    assert mgr.match_device(identity) == []
    assert normalize_tokens("USB Serial") <= GENERIC_TOKENS


def test_corporate_suffix_does_not_mask_brand_word(tmp_path):
    """A vendor name with corporate-suffix words still matches on its
    genuine brand word ("Frobnicate Technology" matches via
    "frobnicate"); a purely corporate name never does."""
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate Technology")
    _make_profile(tmp_path, "generic-co", "Generic Co", "Industry Ltd")
    mgr = _manager(tmp_path)

    identity = _identity_for("Frobnicate laser engraver")
    matches = _matches(mgr, identity)
    assert _names(matches) == ["Frob One"]

    # A purely corporate vendor name never matches anything.
    assert mgr.match_device(_identity_for("Industry laser")) == []


def test_unique_model_identification_is_not_adoptable(tmp_path):
    """Model tokens alone (without the profile's full product name)
    are strong evidence, but not certain identification."""
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate", model="One")
    _make_profile(tmp_path, "other-two", "Other Two", "Gadgetco")
    mgr = _manager(tmp_path)

    identity = _identity_for("Frobnicate One laser")
    matches = _matches(mgr, identity)
    assert _names(matches) == ["Frob One"]
    assert matches[0].confidence == CONFIDENCE_MODEL_TOKENS
    assert certain_match(matches) is None


def test_unique_vendor_only_match_is_not_adoptable(tmp_path):
    _make_profile(tmp_path, "frob-one", "Frob One", "Frobnicate", model="One")
    _make_profile(tmp_path, "frob-two", "Frob Two", "Frobnicate")
    mgr = _manager(tmp_path)

    # Only the vendor token is known: both same-vendor profiles show
    # up at vendor confidence (the missing declared model of
    # "Frob One" does not exclude it), and none is adoptable.
    identity = _identity_for("Frobnicate laser")
    matches = _matches(mgr, identity)
    assert set(_names(matches)) == {"Frob One", "Frob Two"}
    assert all(m.confidence == CONFIDENCE_VENDOR_TOKENS for m in matches)
    assert certain_match(matches) is None


def test_custom_usb_id_is_certain(tmp_path):
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["abcd:1234"],
    )
    _make_profile(tmp_path, "frob-two", "Frob Two", "Frobnicate")
    mgr = _manager(tmp_path)

    # Token matching alone would rank both same-vendor profiles
    # equally; the vendor-specific vid/pid declaration is certain and
    # wins outright.
    identity = _usb_identity(CUSTOM_VID, CUSTOM_PID, "Frobnicate laser")
    matches = _matches(mgr, identity)
    assert _names(matches)[0] == "Frob One"
    assert matches[0].confidence == CONFIDENCE_CERTAIN
    assert certain_match(matches) is not None


def test_generic_bridge_usb_id_is_capped(tmp_path):
    """Stock USB-serial chip ids are shared across unrelated
    products: they yield a low-confidence candidate, never a certain
    match."""
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["1a86:7523"],
    )
    mgr = _manager(tmp_path)

    identity = _usb_identity(0x1A86, 0x7523)
    matches = _matches(mgr, identity)
    assert _names(matches) == ["Frob One"]
    assert matches[0].confidence == CONFIDENCE_GENERIC_USB_ID
    assert certain_match(matches) is None


def test_pid_mismatch_lowers_confidence_but_keeps_profile(tmp_path):
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["abcd:1234"],
    )
    mgr = _manager(tmp_path)

    identity = _usb_identity(CUSTOM_VID, 0xDEAD, "Frobnicate laser")
    matches = _matches(mgr, identity)
    assert _names(matches) == ["Frob One"]
    assert matches[0].confidence == CONFIDENCE_VENDOR_TOKENS
    assert certain_match(matches) is None

    tokenless = _usb_identity(CUSTOM_VID, 0xDEAD)
    assert mgr.match_device(tokenless) == []


def test_vid_only_declaration_matches_any_pid(tmp_path):
    _make_profile(
        tmp_path, "frob-one", "Frob One", "Frobnicate", usb_ids=["abcd"]
    )
    mgr = _manager(tmp_path)

    matched = mgr.match_device(_usb_identity(CUSTOM_VID, 0x9999))
    assert _names(matched) == ["Frob One"]
    assert matched[0].confidence == CONFIDENCE_VID_ONLY
    assert certain_match(matched) is None


def test_any_pid_entry_survives_sibling_specific_pids(tmp_path):
    """A profile declaring both specific pids and an "any product id"
    entry must still match at vid-only confidence when the device's
    pid is not among the specific ones."""
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["abcd:1234", "abcd"],
    )
    mgr = _manager(tmp_path)

    matched = mgr.match_device(_usb_identity(CUSTOM_VID, 0x9999))
    assert _names(matched) == ["Frob One"]
    assert matched[0].confidence == CONFIDENCE_VID_ONLY
    assert certain_match(matched) is None

    # An exact specific pid still wins outright.
    certain = mgr.match_device(_usb_identity(CUSTOM_VID, CUSTOM_PID))
    assert certain[0].profile.name == "Frob One"
    assert certain[0].confidence == CONFIDENCE_CERTAIN


def test_ambiguous_certain_usb_ids_are_not_auto_adopted(tmp_path):
    """Two profiles claiming the same vendor-specific vid/pid stay
    ambiguous."""
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["abcd:1234"],
    )
    _make_profile(
        tmp_path,
        "gadget-two",
        "Gadget Two",
        "Gadgetco",
        usb_ids=["abcd:1234"],
    )
    mgr = _manager(tmp_path)

    matches = mgr.match_device(_usb_identity(CUSTOM_VID, CUSTOM_PID))
    assert len(matches) == 2
    assert all(m.confidence == CONFIDENCE_CERTAIN for m in matches)
    assert certain_match(matches) is None


def test_profiles_without_usb_ids_are_unaffected(tmp_path):
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["0483:5740"],
    )
    _make_profile(tmp_path, "other-two", "Other Two", "Gadgetco")
    mgr = _manager(tmp_path)

    identity = _identity_for("Gadgetco Laser Master", "CH340")
    matches = mgr.match_device(identity)
    assert _names(matches) == ["Other Two"]

    tokenless = DeviceIdentity(usb_vid=0x0483, usb_pid=0x9999)
    assert mgr.match_device(tokenless) == []


def test_journal_records_match_events(tmp_path):
    journal = tmp_path / "journal.jsonl"
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["abcd:1234"],
    )
    mgr = DeviceProfileManager(source_dirs=[tmp_path], journal_file=journal)
    mgr.discover()

    mgr.match_device(_usb_identity(CUSTOM_VID, CUSTOM_PID))
    entry = discovery_journal.read_entries(journal)[0]
    assert entry["usb_vid"] == "abcd"
    assert entry["matches"][0]["profile"] == "Frob One"

    # A manager without a journal file never writes one.
    plain = DeviceProfileManager(source_dirs=[tmp_path])
    plain.discover()
    plain.match_device(_usb_identity(CUSTOM_VID, CUSTOM_PID))
    assert discovery_journal.read_entries(journal)[0] == entry


def test_journal_flag_suppresses_recording(tmp_path):
    """journal=False is for display-only pre-filtering: the event is
    scored but not recorded, so one selection is not journaled twice."""
    journal = tmp_path / "journal.jsonl"
    _make_profile(
        tmp_path,
        "frob-one",
        "Frob One",
        "Frobnicate",
        usb_ids=["abcd:1234"],
    )
    mgr = DeviceProfileManager(source_dirs=[tmp_path], journal_file=journal)
    mgr.discover()
    identity = _usb_identity(CUSTOM_VID, CUSTOM_PID)

    assert mgr.match_device(identity)
    assert len(discovery_journal.read_entries(journal)) == 1

    assert mgr.match_device(identity, journal=False)
    assert len(discovery_journal.read_entries(journal)) == 1


def test_builtin_profiles_parse_without_usb_id_errors():
    devices_dir = Path(rayforge.__file__).parent / "resources" / "devices"
    mgr = DeviceProfileManager(source_dirs=[devices_dir])
    mgr.discover()
    assert mgr.get_load_errors() == {}


def test_invalid_usb_id_raises(tmp_path):
    profile_dir = tmp_path / "bad-usb"
    profile_dir.mkdir()
    (profile_dir / "device.yaml").write_text(
        """\
api_version: 1
device:
  name: Bad Usb
  vendor: Frobnicate
  usb_ids:
    - not-an-id
machine:
  driver: RuidaDriver
"""
    )
    with pytest.raises(ValueError, match="Invalid USB id"):
        DeviceProfile.from_path(profile_dir)


def test_yaml_roundtrip_of_usb_ids(tmp_path):
    manifest = tmp_path / "device.yaml"
    manifest.write_text(
        """\
api_version: 1
device:
  name: Round Trip
  vendor: Frobnicate
  usb_ids:
    - 1a86:7523
    - 0483
    - abcd:*
machine: {}
"""
    )
    meta = parse_meta(yaml.safe_load(manifest.read_text()), manifest)
    assert meta.usb_ids == [
        (0x1A86, 0x7523),
        (0x0483, None),
        (0xABCD, None),
    ]


def test_builtin_profiles_all_declare_vendor():
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
    devices_dir = Path(rayforge.__file__).parent / "resources" / "devices"
    mgr = DeviceProfileManager(source_dirs=[devices_dir])
    mgr.discover()

    identity = build_identity(
        "grbl",
        port_info=None,
        device_name="Sculpfun iCube",
    )
    matches = mgr.match_device(identity)
    assert matches
    assert matches[0].profile.name == "Sculpfun iCube"
    assert certain_match(matches) is not None
