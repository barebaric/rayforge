import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Callable

from rayforge.shared.tasker import task_mgr
from .generator import generate_svg

logger = logging.getLogger(__name__)


class AISvgGeneratorController:
    """Controller for AI SVG generation - pure business logic."""

    def __init__(self):
        self._cancelled = False

    def generate(
        self,
        prompt: str,
        on_success: Callable[[Path], None],
        on_error: Callable[[str], None],
    ) -> None:
        """
        Generate SVG and save to a temporary file.

        Args:
            prompt: The text prompt for SVG generation
            on_success: Callback with path to the temporary SVG file
            on_error: Callback with error message
        """
        self._cancelled = False

        async def do_generate():
            try:
                svg_content, error = await generate_svg(prompt)

                if self._cancelled:
                    return

                if error:
                    task_mgr.schedule_on_main_thread(on_error, error)
                    return

                if not svg_content:
                    task_mgr.schedule_on_main_thread(
                        on_error, "Failed to generate SVG."
                    )
                    return

                temp_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".svg",
                    delete=False,
                    encoding="utf-8",
                )
                temp_file.write(svg_content)
                temp_file.close()

                if self._cancelled:
                    return

                task_mgr.schedule_on_main_thread(
                    on_success, Path(temp_file.name)
                )

            except Exception as e:
                logger.error("Error in generation task: %s", e, exc_info=True)
                if not self._cancelled:
                    task_mgr.schedule_on_main_thread(on_error, str(e))

        future = asyncio.run_coroutine_threadsafe(do_generate(), task_mgr.loop)

        def on_done(f):
            try:
                f.result()
            except Exception as e:
                logger.error("Generation future error: %s", e)
                if not self._cancelled:
                    task_mgr.schedule_on_main_thread(on_error, str(e))

        future.add_done_callback(on_done)

    def cancel(self) -> None:
        """Cancel any ongoing generation."""
        self._cancelled = True


controller = AISvgGeneratorController()
