"""Physical optical model helpers for the burn char pipeline.

Maps a laser's emission wavelength to a coarse absorption band and
returns the material's absorption coefficient for that band, plus the
char-curve (``burn_response``) parameters that drive the stock
shader's burn block.

Three bands cover the three real laser types shipped today (blue
diode ~455 nm, IR/fiber ~1064 nm, CO₂ ~10600 nm) — deliberately
coarser than a full spectrum, matching the data the material YAMLs
carry (see ``physical-burn.md`` step 1).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .color import hex_to_rgba

# Coarse wavelength bands, ordered by representative wavelength. Each
# band is keyed by a short name that appears in material YAML
# ``appearance.absorption`` dicts. The representative wavelength is
# used to map a laser's ``wavelength_nm`` to the nearest band.
BANDS: tuple[tuple[str, float], ...] = (
    ("blue", 455.0),
    ("ir", 1064.0),
    ("co2", 10600.0),
)

# Absorption returned when the material has an ``absorption`` dict but
# the resolved band is not listed. Neutral, so unconfigured bands do
# not silently zero out the burn.
DEFAULT_BAND_ABSORPTION = 0.5

# Absorption returned when the material has no ``absorption`` dict at
# all — full absorption, preserving the pre-physical-model behaviour
# (the burn depends only on fluence, not on a wavelength coefficient).
DEFAULT_ABSORPTION = 1.0

# Default ``burn_response`` parameters, used when a material's
# ``appearance.burn_response`` is absent or incomplete. These are in
# the fluence (J/cm²) domain, calibrated against real-world desktop
# diode behaviour: a 5 W laser at 10 % power and 1000 mm/min deposits
# ~30 J/cm² on the surface and leaves NO visible mark on wood, while
# full power on a typical raster (~75 J/cm² absorbed) marks clearly.
# Charring therefore starts around 35 J/cm² and saturates near
# 125 J/cm². Vector-cut outline fluence is capped at 100 J/cm² (see
# raygeo's OUTLINE_MAX_FLUENCE) so cuts read as a strong dark kerf
# rather than overexposed black. The char colors are black (not the
# laser color): the physical model produces carbonized material, whose
# color is independent of the beam.
DEFAULT_BURN_RESPONSE: dict[str, Any] = {
    "char_threshold": 35.0,
    "char_saturation": 125.0,
    "char_color_low": (0.04, 0.03, 0.02),
    "char_color_high": (0.01, 0.01, 0.01),
}


def wavelength_to_band(wavelength_nm: float) -> str:
    """Map a laser wavelength in nm to the nearest absorption band.

    Returns one of the band names in :data:`BANDS` (``"blue"``,
    ``"ir"``, ``"co2"``). Wavelengths equidistant between two bands
    round to the physically closer representative; the three bands are
    spaced far enough apart (logarithmically) that any real laser
    wavelength lands unambiguously.
    """
    if not wavelength_nm or wavelength_nm <= 0:
        return BANDS[0][0]
    nearest = min(BANDS, key=lambda b: abs(b[1] - wavelength_nm))
    return nearest[0]


def absorption_for(
    wavelength_nm: float, absorption: Mapping[str, float] | None
) -> float:
    """Absorption coefficient (0–1) for a wavelength and material.

    ``absorption`` is the material's ``appearance.absorption`` dict
    (band name → coefficient). ``None`` (no optical data) returns
    :data:`DEFAULT_ABSORPTION` (1.0, full absorption — current
    behaviour). A present dict with the resolved band missing returns
    :data:`DEFAULT_BAND_ABSORPTION` (0.5, neutral).
    """
    if absorption is None:
        return DEFAULT_ABSORPTION
    band = wavelength_to_band(wavelength_nm)
    value = absorption.get(band)
    if value is None:
        return DEFAULT_BAND_ABSORPTION
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return DEFAULT_BAND_ABSORPTION


def _coerce_color(value: Any) -> tuple[float, float, float]:
    """Coerce a YAML color entry to a 0–1 RGB triple.

    Accepts a 3-tuple/list of floats/ints (0–1 or 0–255) or a hex
    string. Falls back to a neutral grey so a malformed entry never
    breaks the shader.
    """
    if isinstance(value, str):
        try:
            r, g, b, _a = hex_to_rgba(value)
            return (r, g, b)
        except ValueError:
            return (0.5, 0.5, 0.5)
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        out = []
        for v in value[:3]:
            try:
                f = float(v)
            except (TypeError, ValueError):
                return (0.5, 0.5, 0.5)
            out.append(f / 255.0 if f > 1.0 else f)
        return (out[0], out[1], out[2])
    return (0.5, 0.5, 0.5)


def burn_response_for(
    burn_response: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolved char-curve parameters for a material.

    Merges the material's ``appearance.burn_response`` (if any) over
    :data:`DEFAULT_BURN_RESPONSE`, coercing color entries to 0–1 RGB
    triples and clamping thresholds to non-negative fluence (J/cm²)
    with ``char_saturation > char_threshold``. The returned dict
    always has all keys, so the shader wiring can index it
    unconditionally.
    """
    out = dict(DEFAULT_BURN_RESPONSE)
    if burn_response:
        for key in ("char_threshold", "char_saturation"):
            if key in burn_response:
                try:
                    out[key] = max(0.0, float(burn_response[key]))
                except (TypeError, ValueError):
                    pass
        for key in ("char_color_low", "char_color_high"):
            if key in burn_response:
                out[key] = _coerce_color(burn_response[key])
    # Ensure a non-degenerate ramp: saturation must exceed threshold.
    if out["char_saturation"] <= out["char_threshold"]:
        out["char_saturation"] = out["char_threshold"] + 1e-3
    return out


def material_absorption(material: Any, wavelength_nm: float) -> float:
    """Absorption for a :class:`~rayforge.core.material.Material`.

    Reads the material's ``appearance.absorption`` extra field. ``None``
    for a material without optical data (the common case) returns
    :data:`DEFAULT_ABSORPTION`.
    """
    absorption = _appearance_extra(material).get("absorption")
    if not isinstance(absorption, Mapping):
        absorption = None
    return absorption_for(wavelength_nm, absorption)


def material_burn_response(material: Any) -> dict[str, Any]:
    """Resolved burn-response dict for a :class:`Material`."""
    br = _appearance_extra(material).get("burn_response")
    if not isinstance(br, dict):
        br = None
    return burn_response_for(br)


def _appearance_extra(material: Any) -> dict[str, Any]:
    """The material's ``appearance.extra`` dict, or ``{}`` if absent."""
    appearance = getattr(material, "appearance", None)
    if appearance is None:
        return {}
    return getattr(appearance, "extra", None) or {}
