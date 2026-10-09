# flake8: noqa: E402
"""UI tests: the Perforation settings group edits cut and skip lengths."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest
from post_processors.transformers import PerforationTransformer
from post_processors.widgets import TRANSFORMER_WIDGETS
from post_processors.widgets.perforation_group import (
    PerforationSettingsGroup,
)


class _Page:
    use_expanders = True


def _build_group():
    transformer = PerforationTransformer(cut_length=1.5, skip_length=0.5)
    group = PerforationSettingsGroup("Perforation", transformer, _Page())
    return group, transformer


@pytest.mark.ui
def test_perforation_group_is_registered(ui_context):
    assert TRANSFORMER_WIDGETS[PerforationTransformer] is (
        PerforationSettingsGroup
    )


@pytest.mark.ui
def test_perforation_group_shows_lengths(ui_context):
    group, _transformer = _build_group()
    assert group.cut_row.get_value_in_base_units() == pytest.approx(1.5)
    assert group.skip_row.get_value_in_base_units() == pytest.approx(0.5)


@pytest.mark.ui
def test_perforation_group_reports_changes(ui_context):
    group, _transformer = _build_group()
    sent = []

    def on_param(sender, key, value, name):
        sent.append((key, value))

    group.param_changed.connect(on_param)
    group.cut_row.set_value_in_base_units(3.0)
    group._on_cut_changed(group.cut_row)
    group.skip_row.set_value_in_base_units(0.25)
    group._on_skip_changed(group.skip_row)

    assert ("cut_length", pytest.approx(3.0)) in sent
    assert ("skip_length", pytest.approx(0.25)) in sent
