"""AI-powered machine specification lookup.

This module queries the configured AI provider for known machine
specifications by vendor + model, returning a structured dictionary
that the unified machine wizard can present as editable suggestion
chips for hardware dimensions, speeds, and head parameters.

The lookup is best-effort: any error (no provider configured, network
failure, unparseable response) results in an empty ``{}`` return so
the wizard falls back gracefully to manual entry.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

from ...context import get_context
from . import AIServiceError
from .provider import ChatMessage

if TYPE_CHECKING:
    from ...context import RayforgeContext

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert assistant for CNC, laser, and 3D printer \
specifications. When asked about a specific machine model, respond ONLY \
with a single JSON object (no markdown fences, no explanation) following \
this exact schema:
{
  "axis_extents": [x_mm, y_mm],
  "max_travel_speed": mm_per_min,
  "max_cut_speed": mm_per_min,
  "acceleration": mm_per_s2,
  "origin": "bottom_left" | "top_left" | "top_right" | "bottom_right",
  "head_type": "laser" | "spindle",
  "max_power": int,           // S-value for laser heads
  "max_rpm": int,             // for spindle heads
  "min_rpm": int,             // for spindle heads
  "spot_size_mm": [x, y],     // for laser heads
  "pwm_frequency": int_hz,    // for laser heads
  "focal_distance": float_mm, // for laser heads
  "home_on_start": bool
}
Omit any field that you do not know with high confidence. Use the \
manufacturer's official specifications when available."""


def _extract_json_object(content: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of an LLM response.

    LLMs occasionally wrap JSON in markdown fences or surrounding
    prose even when asked not to. We try a few strategies in order:
    plain ``json.loads``, then a fenced-codeblock regex, then a
    naive brace-matching scan.
    """
    content = content.strip()
    if not content:
        return None

    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    code_block = re.search(
        r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE
    )
    if code_block:
        try:
            result = json.loads(code_block.group(1).strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    first = content.find("{")
    if first < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(first, len(content)):
        ch = content[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(content[first : i + 1])
                    if isinstance(result, dict):
                        return result
                except json.JSONDecodeError:
                    return None
    return None


def _coerce_specs(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize the parsed JSON into wizard-consumable fields.

    Drops keys with ``None`` values and converts numeric strings to
    floats/ints. Leaves unknown keys intact so the wizard can decide
    what to surface — callers consume defensively.
    """
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                if "." in stripped:
                    coerced: Any = float(stripped)
                else:
                    coerced = int(stripped)
            except ValueError:
                coerced = stripped
        else:
            coerced = value
        out[key] = coerced
    return out


async def lookup_machine_specs(
    vendor: str,
    model: str,
    context: Optional["RayforgeContext"] = None,
) -> dict[str, Any]:
    """Query the AI for machine specifications.

    Args:
        vendor: Manufacturer name (e.g. "Sculpfun").
        model: Machine model (e.g. "S30 Pro").
        context: RayforgeContext. When None, ``get_context()`` is
            used. The default AI provider is queried.

    Returns:
        A normalized dict of spec fields that the wizard knows how
        to surface, or ``{}`` on any error / when no AI provider is
        configured. Callers MUST treat an empty dict as "no info" and
        fall back to manual entry.
    """
    if context is None:
        context = get_context()

    ai_service = context.ai_service
    if not ai_service.get_provider():
        logger.debug("spec_lookup: no AI provider configured, returning empty")
        return {}

    prompt = (
        f"What are the official specifications for the {vendor} "
        f"{model} desktop CNC/laser/3D printer? Include work-area "
        f"dims in mm, max travel and cut speeds in mm/min, "
        f"acceleration, default coordinate origin, head type and "
        f"its key parameters."
    )
    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=prompt),
    ]

    try:
        response = await ai_service.chat(messages)
    except AIServiceError as exc:
        logger.info("spec_lookup: AI service error: %s", exc)
        return {}
    except Exception as exc:
        logger.warning(
            "spec_lookup: unexpected AI error: %s", exc, exc_info=True
        )
        return {}

    if response is None or not response.content:
        logger.debug("spec_lookup: empty AI response")
        return {}

    parsed = _extract_json_object(response.content)
    if parsed is None:
        logger.info(
            "spec_lookup: could not parse JSON from response: %s",
            response.content[:200],
        )
        return {}

    return _coerce_specs(parsed)


__all__ = ["is_ai_configured", "lookup_machine_specs"]


def is_ai_configured(context: Optional["RayforgeContext"] = None) -> bool:
    """Return True when a default AI provider is enabled."""
    if context is None:
        context = get_context()
    return context.ai_service.get_provider() is not None
