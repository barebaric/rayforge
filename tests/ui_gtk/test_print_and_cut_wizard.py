"""Regression tests for Print & Cut wizard position handling (#394).

``Machine.get_current_position()`` returns a variable-length ``Pos``
tuple: 4-axis machines report an extra A-axis value, so the tuple has
four entries. The wizard previously unpacked a fixed 3-tuple, which
raised ``ValueError`` and crashed the wizard on such machines.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REPO_ROOT = Path(__file__).parents[2]
_ADDON_ROOT = (
    _REPO_ROOT / "rayforge" / "builtin_addons" / "rayforge-addon-print-and-cut"
)

# The addon directory name contains hyphens and is not importable the
# normal way, so we mimic the addon manager's synthetic module setup
# (see AddonManager._ensure_parent_modules).
_ROOT_NS = "rayforge_addons"
_ADDON_NS = f"{_ROOT_NS}.print_and_cut"
_INNER_PKG = f"{_ADDON_NS}.print_and_cut"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_wizard_class():
    wizard_mod = sys.modules.get(f"{_INNER_PKG}.wizard")
    if wizard_mod is not None:
        return wizard_mod.PrintAndCutWizard

    root = types.ModuleType(_ROOT_NS)
    root.__path__ = []
    root.__package__ = _ROOT_NS
    sys.modules[_ROOT_NS] = root

    addon = types.ModuleType(_ADDON_NS)
    addon.__path__ = [str(_ADDON_ROOT)]
    addon.__package__ = _ADDON_NS
    sys.modules[_ADDON_NS] = addon

    _load_module(_INNER_PKG, _ADDON_ROOT / "print_and_cut" / "__init__.py")
    wizard_mod = _load_module(
        f"{_INNER_PKG}.wizard", _ADDON_ROOT / "print_and_cut" / "wizard.py"
    )
    return wizard_mod.PrintAndCutWizard


PrintAndCutWizard = _load_wizard_class()


@pytest.fixture
def wizard_stub():
    stub = MagicMock()
    stub._get_pointer_shift.return_value = (0.0, 0.0)
    return stub


@pytest.mark.ui
@pytest.mark.parametrize(
    "pos, expected",
    [
        ((1.0, 2.0, 0.0), "X: 1.00  Y: 2.00"),
        ((1.0, 2.0, 0.0, 45.0), "X: 1.00  Y: 2.00"),
    ],
)
def test_update_laser_position_with_extra_axis(wizard_stub, pos, expected):
    wizard_stub._machine.get_current_position.return_value = pos

    PrintAndCutWizard._update_laser_position(wizard_stub)

    wizard_stub._laser_row.set_subtitle.assert_called_once_with(expected)


@pytest.mark.ui
@pytest.mark.parametrize(
    "pos",
    [
        None,
        (),
        (None, None, None),
        (1.0, None, 0.0, 45.0),
    ],
)
def test_update_laser_position_ignores_invalid_positions(wizard_stub, pos):
    wizard_stub._machine.get_current_position.return_value = pos

    PrintAndCutWizard._update_laser_position(wizard_stub)

    wizard_stub._laser_row.set_subtitle.assert_not_called()


@pytest.mark.ui
@pytest.mark.parametrize(
    "pos",
    [
        (1.5, -2.5, 0.0),
        (1.5, -2.5, 0.0, 45.0),
    ],
)
def test_record_clicked_with_extra_axis(wizard_stub, pos):
    wizard_stub._machine.get_current_position.return_value = pos

    PrintAndCutWizard._on_record_clicked(wizard_stub, None, 0)

    assert wizard_stub._physical_point1 == (1.5, -2.5)
    wizard_stub._pos1_row.set_subtitle.assert_called_once_with("(1.50, -2.50)")


@pytest.mark.ui
@pytest.mark.parametrize(
    "enabled, offset, expected",
    [
        (True, (2.0, -1.0), (2.0, -1.0)),
        (False, (2.0, -1.0), (0.0, 0.0)),
    ],
)
def test_get_pointer_shift_follows_alignment_state(
    wizard_stub, enabled, offset, expected
):
    wizard_stub._machine.pointer_alignment_enabled = enabled
    wizard_stub._machine.get_pointer_offset.return_value = offset

    assert PrintAndCutWizard._get_pointer_shift(wizard_stub) == expected


@pytest.mark.ui
def test_get_pointer_shift_without_machine(wizard_stub):
    wizard_stub._machine = None

    assert PrintAndCutWizard._get_pointer_shift(wizard_stub) == (0.0, 0.0)


@pytest.mark.ui
def test_record_clicked_applies_pointer_shift(wizard_stub):
    wizard_stub._machine.get_current_position.return_value = (10.0, 20.0, 0.0)
    wizard_stub._get_pointer_shift.return_value = (2.0, -1.0)

    PrintAndCutWizard._on_record_clicked(wizard_stub, None, 0)

    assert wizard_stub._physical_point1 == (12.0, 19.0)
    wizard_stub._pos1_row.set_subtitle.assert_called_once_with(
        "(12.00, 19.00)"
    )


@pytest.mark.ui
@pytest.mark.parametrize(
    "point, shift, expected_cmd",
    [
        ((12.0, 19.0), (2.0, -1.0), (10.0, 20.0)),
        ((12.0, 19.0), (0.0, 0.0), (12.0, 19.0)),
    ],
)
def test_goto_clicked_compensates_pointer_shift(
    wizard_stub, point, shift, expected_cmd
):
    wizard_stub._physical_point1 = point
    wizard_stub._get_pointer_shift.return_value = shift

    PrintAndCutWizard._on_goto_clicked(wizard_stub, None, 0)

    wizard_stub._machine_cmd.move_to.assert_called_once_with(
        wizard_stub._machine, *expected_cmd
    )


@pytest.mark.ui
@pytest.mark.parametrize("active", [True, False])
def test_alignment_toggled_updates_machine(wizard_stub, active):
    row = MagicMock()
    row.get_active.return_value = active

    PrintAndCutWizard._on_alignment_toggled(wizard_stub, row, None)

    wizard_stub._machine.set_pointer_alignment.assert_called_once_with(active)


@pytest.mark.ui
def test_update_alignment_row_requires_pointer_offset(wizard_stub):
    wizard_stub._machine.has_pointer_offset.return_value = False

    PrintAndCutWizard._update_alignment_row(wizard_stub)

    wizard_stub._alignment_row.set_sensitive.assert_called_once_with(False)
    wizard_stub._sync_alignment_row_state.assert_called_once_with()


@pytest.mark.ui
def test_update_alignment_row_mirrors_machine_state(wizard_stub):
    wizard_stub._machine.has_pointer_offset.return_value = True
    wizard_stub._machine.pointer_alignment_enabled = True

    PrintAndCutWizard._update_alignment_row(wizard_stub)

    wizard_stub._alignment_row.set_sensitive.assert_called_once_with(True)
    wizard_stub._sync_alignment_row_state.assert_called_once_with()


@pytest.mark.ui
def test_sync_alignment_row_state_mirrors_machine(wizard_stub):
    wizard_stub._machine.pointer_alignment_enabled = True

    PrintAndCutWizard._sync_alignment_row_state(wizard_stub)

    wizard_stub._alignment_row.handler_block.assert_called_once_with(
        wizard_stub._alignment_handler_id
    )
    wizard_stub._alignment_row.set_active.assert_called_once_with(True)
    wizard_stub._alignment_row.handler_unblock.assert_called_once_with(
        wizard_stub._alignment_handler_id
    )
