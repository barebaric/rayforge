"""Confidence-scored matching of discovered devices to profiles.

Instead of returning a single winner, matching assigns every known
profile a confidence in ``[0.0, 1.0]`` describing how sure the
matcher is that the profile describes the discovered device:

* ``1.0`` — certain. Earned by an exact vid/pid match against a
  *vendor-specific* id, or by the device reporting every word of a
  profile's product name (it effectively names itself).
* between 0.0 and 1.0 — plausible candidate. Surfaced in the
  wizard's profile picker, best first, for the user to decide.
* ``0.0`` — no match.

USB ids declared by profiles are positive evidence only: a pid
mismatch lowers confidence but never excludes a profile outright,
because one product line can ship with different USB bridges across
revisions. Stock USB-serial bridge pairs (CH340, CP210x, ...) are
shared across thousands of unrelated products, so they are capped at
:data:`GENERIC_USB_ID_MATCH` and can never produce a certain match.

This module is GTK-free so it can be unit-tested in isolation.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..discovery import (
    CORPORATE_TOKENS,
    GENERIC_TOKENS,
    DeviceIdentity,
    normalize_tokens,
)

if TYPE_CHECKING:
    from .profile import DeviceProfile

#: Fully identified; the wizard may adopt this profile unasked.
CONFIDENCE_CERTAIN = 1.0

#: Exact vendor-specific vid/pid pair declared by the profile.
CONFIDENCE_USB_ID = 0.9

#: Vendor and model text tokens both found in the device identity.
CONFIDENCE_MODEL_TOKENS = 0.8

#: Vendor-only usb id declaration ("any product id") matched.
CONFIDENCE_VID_ONLY = 0.7

#: Vendor text tokens found in the device identity.
CONFIDENCE_VENDOR_TOKENS = 0.6

#: Match against a stock USB-serial bridge chip id; shared by
#: countless unrelated products, hence weak evidence.
CONFIDENCE_GENERIC_USB_ID = 0.4

#: Known stock USB-serial bridge vid/pid pairs (CH340, CH341,
#: CH9102, CP210x, FT232, PL2303). These identify the chip, not the
#: laser, so they never yield :data:`CONFIDENCE_CERTAIN`.
GENERIC_USB_IDS = frozenset(
    {
        (0x1A86, 0x7523),
        (0x1A86, 0x7522),
        (0x1A86, 0x55D4),
        (0x10C4, 0xEA60),
        (0x0403, 0x6001),
        (0x067B, 0x2303),
    }
)


@dataclass(frozen=True)
class ProfileMatch:
    """A candidate profile and how confident the matcher is in it."""

    profile: "DeviceProfile"
    confidence: float


def score_usb_ids(
    usb_ids: Iterable[tuple[int, int | None]],
    identity: DeviceIdentity,
) -> float:
    """Confidence contributed by the profile's declared usb ids."""
    if identity.usb_vid is None:
        return 0.0
    pids = [pid for vid, pid in usb_ids if vid == identity.usb_vid]
    if not pids:
        return 0.0
    specific = {pid for pid in pids if pid is not None}
    if not specific:
        return CONFIDENCE_VID_ONLY
    if identity.usb_pid is None or identity.usb_pid not in specific:
        return 0.0
    pair = (identity.usb_vid, identity.usb_pid)
    if pair in GENERIC_USB_IDS:
        return CONFIDENCE_GENERIC_USB_ID
    return CONFIDENCE_CERTAIN


def score_tokens(
    name: str, vendor: str, model: str, tokens: frozenset[str]
) -> float:
    """Confidence contributed by brand/model text tokens."""
    vendor_tokens = normalize_tokens(vendor) - CORPORATE_TOKENS
    if not vendor_tokens or vendor_tokens <= GENERIC_TOKENS:
        return 0.0
    if not vendor_tokens <= tokens:
        return 0.0
    # Every word of the profile's display name appears in what the
    # device reported about itself — effectively the machine
    # announcing this exact product. Only counts when the model
    # matched too, so a name whose distinctive words were dropped by
    # tokenization cannot masquerade as certainty.
    name_tokens = normalize_tokens(name)
    model_tokens = normalize_tokens(model)
    if (
        model_tokens
        and model_tokens <= tokens
        and name_tokens
        and name_tokens <= tokens
    ):
        return CONFIDENCE_CERTAIN
    if model_tokens:
        if model_tokens <= tokens:
            return CONFIDENCE_MODEL_TOKENS
        return 0.0
    return CONFIDENCE_VENDOR_TOKENS


def score_profile(profile: "DeviceProfile", identity: DeviceIdentity) -> float:
    """Overall confidence that *profile* describes *identity*."""
    meta = profile.meta
    return max(
        score_tokens(meta.name, meta.vendor, meta.model, identity.tokens),
        score_usb_ids(meta.usb_ids, identity),
    )


def certain_match(
    matches: list[ProfileMatch],
) -> ProfileMatch | None:
    """
    The single certain match, if exactly one exists.

    Two profiles reaching :data:`CONFIDENCE_CERTAIN` stay ambiguous
    and must not be adopted automatically.
    """
    top = [m for m in matches if m.confidence >= CONFIDENCE_CERTAIN]
    if len(top) == 1:
        return top[0]
    return None


__all__ = [
    "CONFIDENCE_CERTAIN",
    "CONFIDENCE_GENERIC_USB_ID",
    "CONFIDENCE_MODEL_TOKENS",
    "CONFIDENCE_USB_ID",
    "CONFIDENCE_VENDOR_TOKENS",
    "CONFIDENCE_VID_ONLY",
    "GENERIC_USB_IDS",
    "ProfileMatch",
    "certain_match",
    "score_profile",
    "score_tokens",
    "score_usb_ids",
]
