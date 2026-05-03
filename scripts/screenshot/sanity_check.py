#!/usr/bin/env python3
"""
Screenshot: Sanity check dialog.

Usage: pixi run screenshot sanity-check
"""

import time
import logging
from threading import Event
from rayforge.uiscript import app, win
from utils import (
    load_project,
    wait_for_settled,
    take_screenshot,
    run_on_main_thread,
    set_window_size,
)

logger = logging.getLogger(__name__)


def main():
    set_window_size(win, 1400, 1000)

    load_project(win, "rects.ryp")
    logger.info("Waiting for document to settle...")
    if not wait_for_settled(win, timeout=10):
        logger.error("Document did not settle in time")
        app.quit_idle()
        return

    from rayforge.context import get_context
    from rayforge.machine.sanity.checker import SanityChecker
    from rayforge.machine.sanity.result import CheckMode
    from rayforge.pipeline.artifact.job import JobArtifact
    from rayforge.ui_gtk.shared.sanity_check_dialog import SanityCheckDialog

    config = get_context().config
    machine = config.machine
    if not machine:
        logger.error("No machine configured")
        app.quit_idle()
        return

    pipeline = win.doc_editor.pipeline
    ops_result = {}
    done_event = Event()

    def on_job_done(handle, error):
        if error or not handle:
            ops_result["error"] = error or RuntimeError(
                "No job handle returned"
            )
            done_event.set()
            return
        try:
            am = pipeline.artifact_manager
            with am.checkout_handle(handle) as artifact:
                if isinstance(artifact, JobArtifact):
                    ops_result["ops"] = artifact.ops
                else:
                    ops_result["error"] = RuntimeError(
                        "Not a JobArtifact"
                    )
        except Exception as e:
            ops_result["error"] = e
        done_event.set()

    pipeline.generate_job_artifact(when_done=on_job_done)

    if not done_event.wait(timeout=20.0):
        logger.error("Job generation timed out")
        app.quit_idle()
        return

    if "error" in ops_result:
        logger.error(f"Job generation failed: {ops_result['error']}")
        app.quit_idle()
        return

    if "ops" not in ops_result:
        logger.error("No ops in job result")
        app.quit_idle()
        return

    checker = SanityChecker(machine)
    report = checker.check(ops_result["ops"], mode=CheckMode.FAST)

    logger.info(
        f"Sanity check found {len(report.issues)} issue(s): "
        f"{sum(1 for i in report.issues if i.severity.value == 'error')} errors, "
        f"{sum(1 for i in report.issues if i.severity.value == 'warning')} warnings"
    )

    if report.is_clean:
        logger.warning("No issues found, dialog will look empty")

    def open_dialog():
        dialog = SanityCheckDialog(
            parent=win,
            report=report,
        )
        dialog.set_size_request(600, -1)
        dialog.present()
        return dialog

    dialog = run_on_main_thread(open_dialog)

    time.sleep(1.0)

    logger.info("Taking screenshot: sanity-check.png")
    take_screenshot("sanity-check.png")

    time.sleep(0.25)

    def close_dialog():
        dialog.close()

    run_on_main_thread(close_dialog)
    app.quit_idle()


main()
