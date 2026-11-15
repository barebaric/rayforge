"""
Screenshot: Addon detail dialog.

Usage: pixi run screenshot app-settings:addons:detail
"""

import logging
import time

from utils import (
    restore_config,
    run_on_main_thread,
    take_screenshot,
)

from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


@restore_config
def main():
    time.sleep(0.25)

    from rayforge.context import get_context
    from rayforge.ui_gtk.addon_manager.addon_detail_dialog import (
        AddonDetailDialog,
    )

    def open_dialog():
        am = get_context().addon_mgr
        addon = am.get_installed_addon("sketcher")
        if addon is None:
            addon = next(iter(am.loaded_addons.values()))
        dialog = AddonDetailDialog(
            win,
            addon.metadata,
            on_install=lambda _meta, _update: None,
        )
        dialog.present()
        return dialog

    dialog = run_on_main_thread(open_dialog)

    def first_image_loaded():
        carousel = dialog._carousel
        if carousel is None or carousel.get_n_pages() == 0:
            return False
        slide = carousel.get_nth_page(0)
        return getattr(slide, "_texture", None) is not None

    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if run_on_main_thread(first_image_loaded):
            break
        time.sleep(0.1)
    time.sleep(0.25)

    logger.info("Taking screenshot: app-settings-addons-detail.webp")
    take_screenshot()

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
