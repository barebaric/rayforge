import logging
import re
from gettext import gettext as _
from typing import Optional, Tuple

from rayforge.context import get_context
from rayforge.core.ai import AIServiceError
from rayforge.core.ai.provider import ChatMessage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an expert SVG generator for laser cutting.
When asked to generate a design, output ONLY valid SVG code with no
explanations.

Requirements:
- Output ONLY the SVG code, no markdown, no explanations, no code blocks
- ALWAYS include width and height attributes in millimeters
  (e.g., width="50mm" height="50mm")
- Set width/height to match the requested physical dimensions
- Use a viewBox that matches the width/height values
  (e.g., viewBox="0 0 50 50" for 50mm)
- All shapes must be closed paths suitable for laser cutting
- Use simple stroke="black" for cut lines

Example output for a 50x50mm square:
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 50 50"
     width="50mm" height="50mm">
  <rect x="5" y="5" width="40" height="40" stroke="black"/>
</svg>"""


def extract_svg_from_response(content: str) -> Optional[str]:
    """Extract SVG code from AI response, handling various formats."""
    content = content.strip()

    if content.startswith("<svg"):
        if "</svg>" in content:
            return content[: content.index("</svg>") + 6]
        return content

    code_block_match = re.search(
        r"```(?:svg|xml)?\s*\n(.*?)\n```", content, re.DOTALL | re.IGNORECASE
    )
    if code_block_match:
        return code_block_match.group(1).strip()

    svg_match = re.search(
        r"<svg[^>]*>.*?</svg>", content, re.DOTALL | re.IGNORECASE
    )
    if svg_match:
        return svg_match.group(0)

    return None


async def generate_svg(prompt: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate SVG from a text prompt using the configured AI provider.

    Returns:
        Tuple of (svg_content, error_message)
    """
    context = get_context()
    ai_service = context.ai_service

    if not ai_service.get_provider():
        return None, _(
            "No AI provider configured. "
            "Please configure an AI provider in Settings."
        )

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(
            role="user", content=f"Generate an SVG design for: {prompt}"
        ),
    ]

    try:
        response = await ai_service.chat(messages)
        if not response:
            return None, _("No response from AI provider.")

        svg_content = extract_svg_from_response(response.content)
        if not svg_content:
            logger.warning(
                "AI response did not contain valid SVG: %s",
                response.content[:200],
            )
            return None, _(
                "AI did not generate valid SVG code. "
                "Please try a different prompt."
            )

        return svg_content, None

    except AIServiceError as e:
        logger.error("AI service error generating SVG: %s", e)
        return None, str(e)
    except Exception as e:
        logger.error("Error generating SVG: %s", e, exc_info=True)
        return None, str(e)
