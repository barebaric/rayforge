"""
Screenshot: App settings - Tools page.

Usage: pixi run screenshot app-settings:tools
"""

import logging
import time

from utils import (
    open_app_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "tools"

SAMPLE_TOOLS = (
    ("6 mm Flat End Mill", "END_MILL", 6.0),
    ("3 mm Ball Nose", "BALL_NOSE", 3.0),
    ("90 deg V-Bit", "VBIT", 6.0),
)


def _seed_tools():
    from rayforge.context import get_context

    get_context().addon_mgr.enable_addon("tool_library")

    from rayforge_addons.tool_library.tool_library import get_tool_manager
    from rayforge_addons.tool_library.tool_library.tool import (
        CATEGORY_BY_NAME,
        Tool,
    )
    from raygeo.cnc.tool import ToolModel

    manager = get_tool_manager()
    for name, category, diameter in SAMPLE_TOOLS:
        tool = Tool.create_default(name)
        tool.category = CATEGORY_BY_NAME[category]
        tool.model = ToolModel(
            diameter=diameter,
            shank_diameter=diameter,
            cutting_edge_height=15.0,
            flute_count=3.0,
            overall_length=50.0,
        )
        manager.save(tool)


def _show_tools_page(dialog):
    from rayforge_addons.tool_library.tool_library.frontend import (
        ToolManagerPage,
    )

    pages = dialog.content_stack.get_pages()
    for index in range(pages.get_n_items()):
        child = pages.get_item(index).get_child()
        if isinstance(child, ToolManagerPage):
            row = dialog.sidebar_list.get_row_at_index(index)
            dialog.sidebar_list.select_row(row)
            return


@restore_config
def main():
    time.sleep(0.25)
    run_on_main_thread(_seed_tools)
    dialog = open_app_settings(win, PAGE)
    run_on_main_thread(lambda: _show_tools_page(dialog))
    time.sleep(0.25)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
