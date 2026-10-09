"""
The pre-flight sanity check must not reuse a cached job when the job
is placed at the current head position: the head may have moved since
that job was assembled.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from rayforge.core.doc import Doc
from rayforge.core.job_origin import JobAnchor, JobOrigin, StartFrom


@pytest.fixture
def window(monkeypatch):
    from rayforge.ui_gtk import mainwindow as mainwindow_module
    from rayforge.ui_gtk.mainwindow import MainWindow

    config = MagicMock()
    monkeypatch.setattr(
        mainwindow_module, "get_context", lambda: MagicMock(config=config)
    )
    win: Any = MainWindow.__new__(MainWindow)
    win.doc_editor = MagicMock()
    win.doc_editor.doc = Doc()
    win.doc_editor.pipeline.get_existing_job_handle.return_value = object()
    return win


@pytest.mark.ui
@pytest.mark.parametrize(
    "start_from,reuses_cache",
    [
        (StartFrom.ABSOLUTE, True),
        (StartFrom.USER_ORIGIN, True),
        (StartFrom.CURRENT_POSITION, False),
    ],
)
def test_sanity_check_cache_use(window, start_from, reuses_cache):
    window.doc_editor.doc.set_job_origin(
        JobOrigin(start_from, JobAnchor.CENTER)
    )
    window._run_sanity_check_and_proceed(MagicMock())
    assemble = window.doc_editor.file.assemble_job_in_background
    assert assemble.called is not reuses_cache
