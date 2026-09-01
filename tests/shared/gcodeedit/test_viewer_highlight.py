# flake8: noqa: E402
"""UI tests for GcodeViewer op highlighting."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

from rayforge.pipeline.encoder.base import MachineCodeOpMap
from rayforge.shared.gcodeedit.viewer import GcodeViewer


@pytest.mark.ui
def test_highlight_op_selects_action_line(ui_context_initializer):
    """Regression test for issue #370.

    The encoder defers state lines like "M4 S800" into the span of the
    first cutting move.  Highlighting an op must select the op's own
    action line (the last of its span), not the deferred state line.
    """
    viewer = GcodeViewer()
    viewer.set_gcode("G21\nG90\nG54\nT0\nG0 X1 Y1\nM4 S800\nG1 X2 Y2 F500\n")
    # Op spans: preamble 3 lines, T0, travel, then the first cut op
    # owns both the deferred "M4 S800" and its own move line.
    viewer.set_op_map(
        MachineCodeOpMap.from_lists(
            [(0, 3), (3, 0), (3, 0), (3, 0), (3, 1), (4, 1), (5, 2), (7, 1)],
            [0, 0, 0, 3, 4, 6, 6, 7],
        )
    )
    # Stepping onto op 6 (the first cutting segment) must highlight
    # line 6 ("G1 X2 Y2 F500"), not line 5 ("M4 S800").
    viewer.highlight_op(6)
    assert viewer.editor.current_highlight_line == 6
    # A single-line op highlights that line (op 4 -> "T0", line 3).
    viewer.highlight_op(4)
    assert viewer.editor.current_highlight_line == 3
    # An op without output clears the highlight.
    viewer.highlight_op(1)
    assert viewer.editor.current_highlight_line == -1
