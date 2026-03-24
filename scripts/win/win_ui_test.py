#!/usr/bin/env python3
"""
Smoke test for Windows executable UI.

This script is run via --uiscript to verify that the application
starts and the main window is visible.
"""

import logging
import sys
import time
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting UI smoke test...")

    if win is None:
        logger.error("FAIL: Main window is None")
        sys.exit(1)

    logger.info("Checking main window visibility...")
    if not win.is_visible():
        logger.error("FAIL: Main window is not visible")
        sys.exit(1)

    logger.info("Checking document editor...")
    _ = win.doc_editor

    logger.info("SUCCESS: UI smoke test passed")
    time.sleep(0.5)
    app.quit_idle()


main()
