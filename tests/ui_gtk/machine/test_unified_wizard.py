"""UI tests for the Unified Machine Configuration Wizard."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from gi.repository import Adw

from rayforge.machine.device.profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)
from rayforge.machine.driver.discovery import DeviceIdentity
from rayforge.machine.driver.grbl import GrblSerialDriver
from rayforge.machine.models.machine import Origin
from rayforge.shared.util.permissions import PermissionIssue
from rayforge.ui_gtk.machine.unified_wizard import UnifiedWizard
from rayforge.ui_gtk.machine.wizard_pages.ai_lookup_page import AILookupPage
from rayforge.ui_gtk.machine.wizard_pages.camera_page import CameraPage
from rayforge.ui_gtk.machine.wizard_pages.connection_page import ConnectionPage
from rayforge.ui_gtk.machine.wizard_pages.controller_page import ControllerPage
from rayforge.ui_gtk.machine.wizard_pages.discover_page import DiscoverPage
from rayforge.ui_gtk.machine.wizard_pages.permissions_page import (
    PermissionsPage,
)
from rayforge.ui_gtk.machine.wizard_pages.probe_page import ProbePage
from rayforge.ui_gtk.machine.wizard_pages.profile_page import ProfilePage
from rayforge.ui_gtk.machine.wizard_pages.provider_page import AIProviderPage
from rayforge.ui_gtk.machine.wizard_pages.review_page import ReviewPage

_CAMERA_PATCH_TARGET = (
    "rayforge.ui_gtk.machine.wizard_pages.camera_page.get_sorted_by_id_paths"
)

_PERMISSIONS_PATCH_TARGET = (
    "rayforge.ui_gtk.machine.unified_wizard.check_permissions"
)
_DISCOVER_PATCH_TARGET = (
    "rayforge.ui_gtk.machine.wizard_pages.discover_page.find_all_devices"
)


@pytest.fixture(autouse=True)
def _inert_device_discovery():
    """Wizard construction opens the discover page, which would start
    scanning real serial ports, and runs filesystem-only permission
    checks whose outcome depends on the host. Keep every test inert."""

    async def _fake_find_all_devices(driver_classes=None, **kwargs):
        return []

    with (
        patch(_DISCOVER_PATCH_TARGET, _fake_find_all_devices),
        patch(_PERMISSIONS_PATCH_TARGET, return_value=[]),
    ):
        yield


def _profile(driver=None):
    return DeviceProfile(
        meta=DeviceMeta(name="Test Machine"),
        machine_config=MachineConfig(driver=driver),
        dialect_config={},
    )


def _make_wizard(ui_context_initializer):
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.get_context",
        return_value=ui_context_initializer,
    ):
        return UnifiedWizard()


@pytest.mark.ui
def test_wizard_starts_on_discover_page(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    assert wizard.stack.get_visible_child_name() == "discover"


def _serial_issue():
    return PermissionIssue(
        category="serial",
        title="Serial Port Access",
        summary="Serial ports cannot be opened.",
        commands=["sudo usermod -a -G dialout $USER"],
        note="Log out and back in.",
    )


@pytest.mark.ui
def test_wizard_starts_on_permissions_page_when_issues(ui_context_initializer):
    issues = [_serial_issue()]
    with patch(_PERMISSIONS_PATCH_TARGET, return_value=issues):
        wizard = _make_wizard(ui_context_initializer)
    assert wizard.stack.get_visible_child_name() == "permissions"
    # The pre-flight page never blocks the flow.
    assert wizard.next_btn.get_visible()
    assert wizard.next_btn.get_sensitive()
    assert not wizard.skip_btn.get_visible()


@pytest.mark.ui
def test_permissions_next_routes_to_discover(ui_context_initializer):
    issues = [_serial_issue()]
    with patch(_PERMISSIONS_PATCH_TARGET, return_value=issues):
        wizard = _make_wizard(ui_context_initializer)
    assert wizard._next_step_after("permissions") == "discover"
    wizard._on_next_clicked(wizard.next_btn)
    assert wizard.stack.get_visible_child_name() == "discover"


@pytest.mark.ui
def test_permissions_page_never_shows_back(ui_context_initializer):
    """The pre-flight page opens the wizard; Back has no meaning there,
    even when reached again via the history stack."""
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("permissions")
    wizard._history.append("discover")
    wizard._update_footer("permissions", wizard._get_page("permissions"))
    assert not wizard.back_btn.get_visible()


@pytest.mark.ui
def test_permissions_page_lists_copyable_commands(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("permissions")
    page = wizard._get_page("permissions")
    assert isinstance(page, PermissionsPage)

    issue = PermissionIssue(
        category="serial",
        title="Serial Port Access",
        summary="Serial ports cannot be opened.",
        commands=[
            "sudo snap set system experimental.hotplug=true",
            "sudo snap connect rayforge:serial-port",
        ],
    )
    page._rebuild([issue])

    assert [r._command for r in page._command_rows] == [
        "sudo snap set system experimental.hotplug=true",
        "sudo snap connect rayforge:serial-port",
    ]
    assert page.ready is True


@pytest.mark.ui
def test_permissions_page_shows_ok_state_when_clear(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("permissions")
    page = wizard._get_page("permissions")
    assert isinstance(page, PermissionsPage)

    page._rebuild([])
    assert page._command_rows == []
    assert page.intro_label.get_text() != ""


@pytest.mark.ui
def test_initial_step_skips_permissions_when_check_fails(
    ui_context_initializer,
):
    """A broken permission check must never trap the user."""
    with patch(_PERMISSIONS_PATCH_TARGET, side_effect=RuntimeError("boom")):
        wizard = _make_wizard(ui_context_initializer)
    assert wizard._initial_step() == "discover"


@pytest.mark.ui
def test_button_bar_lives_inside_main_box(ui_context_initializer):
    """The footer button bar must sit inside the main box, directly
    under the stack, matching the camera calibration wizard layout."""
    wizard = _make_wizard(ui_context_initializer)
    assert wizard._button_box.get_parent() is wizard._main_box
    assert wizard.stack.get_next_sibling() is wizard._button_box


@pytest.mark.ui
def test_footer_action_buttons_follow_page(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)

    discover_page = wizard._get_page("discover")
    assert discover_page is not None
    assert [b.get_label() for b in wizard._footer_action_buttons] == [
        b.get_label() for b in discover_page.footer_buttons()
    ]

    wizard._navigate_to("profile")
    profile_page = wizard._get_page("profile")
    assert profile_page is not None
    assert [b.get_label() for b in wizard._footer_action_buttons] == [
        b.get_label() for b in profile_page.footer_buttons()
    ]

    wizard._navigate_to("ai_lookup")
    ai_page = wizard._get_page("ai_lookup")
    assert ai_page is not None
    assert [b.get_label() for b in wizard._footer_action_buttons] == [
        b.get_label() for b in ai_page.footer_buttons()
    ]

    wizard._navigate_to("profile")
    assert [b.get_label() for b in wizard._footer_action_buttons] == [
        b.get_label() for b in profile_page.footer_buttons()
    ]


@pytest.mark.ui
def test_probe_page_footer_buttons(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("probe")
    labels = [b.get_label() for b in wizard._footer_action_buttons]
    assert labels == ["Probe Now"]

    page = wizard._get_page("probe")
    assert isinstance(page, ProbePage)
    page._show_error("Connection refused")
    # The single probe button relabels to "Retry"; there must never
    # be a redundant "Probe Now" + "Retry" pair.
    assert [b.get_label() for b in wizard._footer_action_buttons] == ["Retry"]


@pytest.mark.ui
def test_wizard_step_order_declared():
    from rayforge.ui_gtk.machine.unified_wizard import _STEP_ORDER

    assert _STEP_ORDER == [
        "permissions",
        "discover",
        "profile",
        "controller",
        "connect",
        "probe",
        "ai_provider",
        "ai_lookup",
        "hardware",
        "head",
        "rotary",
        "camera",
        "review",
    ]


@pytest.mark.ui
def test_route_controller_with_driver_goes_to_connection(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver="GrblSerialDriver")
    assert wizard._next_step_after("controller") == "connect"


@pytest.mark.ui
def test_route_controller_without_driver_skips_connection(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver=None)
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=True,
    ):
        assert wizard._next_step_after("controller") == "ai_lookup"
    assert {"connect", "probe"} <= wizard._skipped_steps_set


@pytest.mark.ui
def test_route_controller_without_driver_no_ai_goes_to_provider(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver=None)
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=False,
    ):
        assert wizard._next_step_after("controller") == "ai_provider"
    assert {"connect", "probe"} <= wizard._skipped_steps_set


@pytest.mark.ui
def test_route_connection_probing_driver_goes_to_probe(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver="GrblSerialDriver")
    assert wizard._next_step_after("connect") == "probe"


@pytest.mark.ui
def test_route_connection_non_probing_driver_skips_probe(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver="RuidaDriver")
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=True,
    ):
        assert wizard._next_step_after("connect") == "ai_lookup"
    assert "probe" in wizard._skipped_steps_set


@pytest.mark.ui
def test_route_ai_provider_next_goes_to_lookup_when_configured(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=True,
    ):
        assert wizard._next_step_after("ai_provider") == "ai_lookup"


@pytest.mark.ui
def test_route_ai_provider_next_goes_to_hardware_when_unconfigured(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=False,
    ):
        assert wizard._next_step_after("ai_provider") == "hardware"


@pytest.mark.ui
def test_route_tail_steps_are_linear(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=True,
    ):
        assert wizard._next_step_after("probe") == "ai_lookup"
        assert wizard._next_step_after("ai_lookup") == "hardware"
    assert wizard._next_step_after("hardware") == "head"
    assert wizard._next_step_after("head") == "rotary"
    assert wizard._next_step_after("rotary") == "camera"
    assert wizard._next_step_after("camera") == "review"


@pytest.mark.ui
def test_route_unknown_step_returns_none(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    assert wizard._next_step_after("review") is None


@pytest.mark.ui
def test_source_selected_known_profile_goes_to_connection(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard._on_profile_source_selected(
        None, kind="profile", profile=_profile()
    )
    assert wizard.stack.get_visible_child_name() == "connect"
    assert "controller" in wizard._skipped_steps_set


@pytest.mark.ui
def test_source_selected_other_goes_to_controller(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._on_profile_source_selected(None, kind="other", profile=None)
    assert wizard.stack.get_visible_child_name() == "controller"


@pytest.mark.ui
def test_clone_profile_preserves_all_config_fields(ui_context_initializer):
    """The working-profile clone must carry every MachineConfig field
    (e.g. has_z) and the stable meta id, so machines created from the
    wizard match direct profile creation."""
    from dataclasses import fields as dc_fields

    src = _profile()
    src.meta.id = "test-machine"
    src.machine_config.has_z = False
    src.machine_config.max_cut_speed = 4242

    wizard = _make_wizard(ui_context_initializer)
    cloned = wizard._clone_profile(src)

    for f in dc_fields(MachineConfig):
        assert getattr(cloned.machine_config, f.name) == getattr(
            src.machine_config, f.name
        ), f"clone dropped field '{f.name}'"
    assert cloned.meta.id == "test-machine"
    assert cloned.dialect_config == src.dialect_config
    assert isinstance(cloned.meta, DeviceMeta)
    # The review page renames the machine via profile.meta.name.
    cloned.meta.name = "Renamed Machine"


@pytest.mark.ui
def test_known_profile_skips_to_review(ui_context_initializer):
    """A picked profile carries trusted specs and optional hardware,
    so after Connection the wizard skips probe, AI, hardware, head,
    rotary and camera — landing on Review. Imports are not fully
    reliable, so they keep hardware/head (but still skip the AI
    pages)."""
    wizard = _make_wizard(ui_context_initializer)
    wizard._on_profile_source_selected(
        None, kind="profile", profile=_profile(driver="RuidaDriver")
    )
    assert wizard._next_step_after("connect") == "review"
    assert {
        "controller",
        "probe",
        "ai_provider",
        "ai_lookup",
        "hardware",
        "head",
        "rotary",
        "camera",
    } <= wizard._skipped_steps_set


@pytest.mark.ui
def test_import_skips_ai_but_keeps_probe_hw_head(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._on_profile_source_selected(
        None, kind="import", profile=_profile(driver="GrblSerialDriver")
    )
    # Import with a probing driver still offers the probe step so the
    # user can verify the imported values.
    assert wizard._next_step_after("connect") == "probe"
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=False,
    ):
        assert wizard._next_step_after("probe") == "hardware"
    assert {"ai_provider", "ai_lookup"} <= wizard._skipped_steps_set
    assert not ({"hardware", "head"} <= wizard._skipped_steps_set)


@pytest.mark.ui
def test_other_source_shows_ai_pages(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._on_profile_source_selected(None, kind="other", profile=None)
    with patch(
        "rayforge.ui_gtk.machine.unified_wizard.is_ai_configured",
        return_value=False,
    ):
        assert wizard._next_step_after("probe") == "ai_provider"


@pytest.mark.ui
def test_ai_lookup_page_content_survives_progress_bar_install(
    ui_context_initializer,
):
    """Regression: the pulse bar wrapper must not orphan the page's
    scrollable content (GTK 'child already has a parent' critical)."""
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("ai_lookup")
    page = wizard._get_page("ai_lookup")
    assert isinstance(page, AILookupPage)
    assert page.vendor_row.get_parent() is not None
    assert page.lookup_button.get_parent() is not None


@pytest.mark.ui
def test_materialize_machine_emits_profile_created(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    created = MagicMock()
    wizard.profile_created.connect(created)

    wizard.profile = _profile()
    # The review page prefills its name row in enter(); navigate there
    # first so apply_to_profile() passes validation.
    wizard._navigate_to("review")
    wizard.create_btn.emit("clicked")

    created.assert_called_once()
    _args, kwargs = created.call_args
    assert kwargs["profile"] is wizard.profile
    assert kwargs["machine"] is not None


@pytest.mark.ui
def test_profile_page_hides_next_button(ui_context_initializer):
    """The profile step advances only via explicit source selection,
    so the generic Next button must not appear next to "Other /
    Unknown Device…" — otherwise the two look like duplicate primary
    actions."""
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("profile")
    assert not wizard.next_btn.get_visible()


@pytest.mark.ui
def test_back_visible_and_returns_to_profile_after_source(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("profile")
    wizard._on_profile_source_selected(
        None, kind="profile", profile=_profile()
    )
    assert wizard.stack.get_visible_child_name() == "connect"
    assert wizard.back_btn.get_visible()
    wizard._on_back_clicked(None)
    assert wizard.stack.get_visible_child_name() == "profile"


@pytest.mark.ui
def test_back_visible_after_other_source(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("profile")
    wizard._on_profile_source_selected(None, kind="other", profile=None)
    assert wizard.stack.get_visible_child_name() == "controller"
    assert wizard.back_btn.get_visible()
    wizard._on_back_clicked(None)
    assert wizard.stack.get_visible_child_name() == "profile"


@pytest.mark.ui
def test_controller_page_requires_explicit_selection(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("controller")
    page = wizard._get_page("controller")
    assert isinstance(page, ControllerPage)
    assert page.ready is False
    assert not wizard.next_btn.get_sensitive()

    # Selecting a controller tile advances instantly (no separate
    # "Next" step).
    page._select_child(page._tiles[0])
    assert wizard.stack.get_visible_child_name() == "connect"
    assert wizard.profile.machine_config.driver is not None


@pytest.mark.ui
def test_controller_page_none_tile_applies_no_driver(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("controller")
    page = wizard._get_page("controller")
    assert isinstance(page, ControllerPage)
    page._select_child(page._tiles[-1])
    # None controller skips connection; routing falls through to AI
    # entry (provider/lookup) per adaptive rules.
    assert wizard.profile.machine_config.driver is None


@pytest.mark.ui
def test_controller_page_enter_preselects_matching_driver(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver="RuidaDriver")
    wizard._navigate_to("controller")
    page = wizard._get_page("controller")
    assert page is not None
    assert page.ready is True
    profile = _profile()
    assert page.apply_to_profile(profile)
    assert profile.machine_config.driver == "RuidaDriver"


@pytest.mark.ui
def test_connection_page_next_requires_details(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard.profile = _profile(driver="OctoPrintDriver")
    wizard._navigate_to("connect")
    page = wizard._get_page("connect")
    assert isinstance(page, ConnectionPage)
    assert page.ready is False
    assert not wizard.next_btn.get_sensitive()

    page.connect_widget.set_values(
        {"host": "octoprint.local", "api_key": "secret"}
    )
    page._refresh_ready()
    assert page.ready is True
    assert wizard.next_btn.get_sensitive()


@pytest.mark.ui
def test_skip_button_only_on_optional_pages(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    for name in ("ai_provider", "rotary", "camera"):
        wizard._navigate_to(name)
        assert wizard.skip_btn.get_visible(), name
    for name in (
        "discover",
        "profile",
        "connect",
        "probe",
        "ai_lookup",
        "hardware",
        "head",
        "review",
    ):
        wizard._navigate_to(name)
        assert not wizard.skip_btn.get_visible(), name


@pytest.mark.ui
def test_ai_provider_page_ready_gated_on_essential_fields(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("ai_provider")
    page = wizard._get_page("ai_provider")
    assert isinstance(page, AIProviderPage)
    assert page.ready is False
    assert not wizard.next_btn.get_sensitive()

    page.api_key_row.set_text("sk-test")
    assert page.ready is True
    assert wizard.next_btn.get_sensitive()


@pytest.mark.ui
def test_ai_provider_page_applies_configuration(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("ai_provider")
    page = wizard._get_page("ai_provider")
    assert isinstance(page, AIProviderPage)
    page.api_key_row.set_text("sk-test")

    with patch(
        "rayforge.ui_gtk.machine.wizard_pages.provider_page.get_context"
    ) as mock_get_context:
        svc = MagicMock()
        mock_get_context.return_value = MagicMock(ai_service=svc)
        profile = _profile()
        assert page.apply_to_profile(profile)
    svc.add_provider.assert_called_once()
    config = svc.add_provider.call_args.args[0]
    assert config.base_url == "https://api.openai.com/v1"
    assert config.api_key == "sk-test"


@pytest.mark.ui
def test_camera_page_selected_device_ids(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("camera")
    page = wizard._get_page("camera")
    assert isinstance(page, CameraPage)

    fake_devices = [
        "/dev/v4l/by-id/usb-fake_cam_0",
        "/dev/v4l/by-id/usb-fake_cam_1",
    ]
    with patch(
        _CAMERA_PATCH_TARGET,
        return_value=fake_devices,
    ):
        page.enter(_profile())

    assert page.selected_device_ids() == []

    switches = [r for r in page._switch_rows if isinstance(r, Adw.SwitchRow)]
    assert len(switches) == 2
    switches[1].set_active(True)
    assert page.selected_device_ids() == [fake_devices[1]]


@pytest.mark.ui
def test_after_camera_next_launches_workflow_for_first_enabled(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("camera")
    page = wizard._get_page("camera")
    assert isinstance(page, CameraPage)

    fake_devices = [
        "/dev/v4l/by-id/usb-fake_cam_0",
        "/dev/v4l/by-id/usb-fake_cam_1",
    ]
    with patch(
        _CAMERA_PATCH_TARGET,
        return_value=fake_devices,
    ):
        page.enter(_profile())
    switches = [r for r in page._switch_rows if isinstance(r, Adw.SwitchRow)]
    switches[0].set_active(True)
    switches[1].set_active(True)

    with patch.object(wizard, "_launch_camera_workflow") as launch:
        wizard._after_camera_next()
    assert wizard.stack.get_visible_child_name() == "review"
    launch.assert_called_once_with(fake_devices[0])


@pytest.mark.ui
def test_after_camera_next_skips_workflow_when_none_enabled(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("camera")
    page = wizard._get_page("camera")
    assert isinstance(page, CameraPage)
    with patch(
        _CAMERA_PATCH_TARGET,
        return_value=[],
    ):
        page.enter(_profile())

    with patch.object(wizard, "_launch_camera_workflow") as launch:
        wizard._after_camera_next()
    assert wizard.stack.get_visible_child_name() == "review"
    launch.assert_not_called()


@pytest.mark.ui
def test_ai_lookup_page_captures_vendor_model(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("ai_lookup")
    page = wizard._get_page("ai_lookup")
    assert isinstance(page, AILookupPage)
    page.vendor_row.set_text("Sculpfun")
    page.model_row.set_text("S30 Pro")
    profile = _profile()
    assert page.apply_to_profile(profile)
    assert profile.meta.vendor == "Sculpfun"
    assert profile.meta.model == "S30 Pro"


@pytest.mark.ui
def test_ai_lookup_page_suggestion_toggles_control_acceptance(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("ai_lookup")
    page = wizard._get_page("ai_lookup")
    assert isinstance(page, AILookupPage)

    specs = {
        "axis_extents": (500, 300),
        "max_cut_speed": 1000,
        "home_on_start": True,
    }
    page._render_suggestions(specs)
    assert set(page._accepted) == set(specs)
    assert len(page._rows) == 3

    off_row = next(r for r in page._rows if "cut speed" in r.get_title())
    off_row.set_active(False)
    assert "max_cut_speed" not in page._accepted

    profile = _profile()
    assert page.apply_to_profile(profile)
    assert profile.machine_config.axis_extents == (500.0, 300.0)
    assert profile.machine_config.max_cut_speed is None
    assert profile.machine_config.home_on_start is True


@pytest.mark.ui
def test_review_page_prefills_name_from_vendor_model(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile()
    profile.meta.name = "New Machine"
    profile.meta.vendor = "Sculpfun"
    profile.meta.model = "S30 Pro"
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    assert isinstance(page, ReviewPage)
    assert page.name_row.get_text() == "Sculpfun S30 Pro"


@pytest.mark.ui
def test_review_page_keeps_explicit_name(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile()
    profile.meta.name = "My Rig"
    profile.meta.vendor = "Sculpfun"
    profile.meta.model = "S30 Pro"
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    assert isinstance(page, ReviewPage)
    assert page.name_row.get_text() == "My Rig"


def _summary_subtitle(page, title):
    for row in page._summary_rows:
        if row.get_title() == title:
            return row.get_subtitle()
    return None


@pytest.mark.ui
def test_review_summary_shows_origin_label(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile()
    profile.machine_config.origin = Origin.BOTTOM_LEFT
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    assert _summary_subtitle(page, "Origin") == "Bottom Left"


@pytest.mark.ui
def test_review_summary_home_on_start_defaults_to_no(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile()
    profile.machine_config.home_on_start = None
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    assert _summary_subtitle(page, "Home on Start") == "No"


@pytest.mark.ui
def test_review_summary_formats_connection_args(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile(driver="GrblSerialDriver")
    profile.machine_config.driver_args = {
        "port": "/dev/ttyUSB0",
        "api_key": "secret123",
    }
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    subtitle = _summary_subtitle(page, "Connection")
    assert subtitle == "Port: /dev/ttyUSB0, api_key: ••••••••"
    assert "{" not in subtitle


@pytest.mark.ui
def test_review_summary_connection_bool_args_translated(
    ui_context_initializer,
):
    """Boolean driver args render as Yes/No, not English True/False."""
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile(driver="GrblSerialDriver")
    profile.machine_config.driver_args = {
        "poll_status_while_running": True,
        "deadlock_detection": False,
    }
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    subtitle = _summary_subtitle(page, "Connection")
    assert subtitle is not None
    assert "Yes" in subtitle
    assert "No" in subtitle
    assert "True" not in subtitle
    assert "False" not in subtitle


@pytest.mark.ui
def test_review_summary_connection_empty_when_no_args(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    profile = _profile(driver="GrblSerialDriver")
    profile.machine_config.driver_args = None
    wizard.profile = profile
    wizard._navigate_to("review")
    page = wizard._get_page("review")
    assert _summary_subtitle(page, "Connection") == "—"


@pytest.mark.ui
def test_ai_lookup_page_progress_bar_tracks_lookup(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard._navigate_to("ai_lookup")
    page = wizard._get_page("ai_lookup")
    assert isinstance(page, AILookupPage)
    assert page._progress_bar is not None
    assert not page._progress_bar.get_visible()
    page._start_pulse()
    assert page._progress_bar.get_visible()
    page._stop_pulse()
    assert not page._progress_bar.get_visible()


def _discovered_device(**overrides):
    from rayforge.machine.driver.discovery import DiscoveredDevice

    defaults = {
        "driver_name": "GrblSerialDriver",
        "params": {"port": "/dev/ttyUSB0", "baudrate": 115200},
        "label": "GRBL device",
        "detail": "/dev/ttyUSB0 at 115200 baud",
        "identity": DeviceIdentity(firmware="grbl", banner="Grbl 1.1f"),
    }
    defaults.update(overrides)
    return DiscoveredDevice(**defaults)


@pytest.mark.ui
def test_discover_page_lists_devices(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    assert wizard.stack.get_visible_child_name() == "discover"
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    page._update_devices([])
    assert page.status_label.get_text() == "No devices detected yet"
    assert page.status_box.get_visible()

    with patch.object(GrblSerialDriver, "supports_probing", False):
        page._update_devices([_discovered_device()])
    assert not page.status_box.get_visible()
    assert list(page._device_rows) == ["GrblSerialDriver:/dev/ttyUSB0"]
    row = page._device_rows["GrblSerialDriver:/dev/ttyUSB0"]
    assert row.get_title() == "GRBL device"
    subtitle = row.get_subtitle()
    assert subtitle is not None
    assert "/dev/ttyUSB0 at 115200 baud" in subtitle

    # A held device is not rescanned, so its row survives empty scan
    # results; only unplugging (port gone from the system) removes it.
    page._update_devices([])
    assert list(page._device_rows) == ["GrblSerialDriver:/dev/ttyUSB0"]


@pytest.mark.ui
def test_discover_page_select_button(ui_context_initializer):
    """Every device row carries a Select button; clicking it picks
    the device just like activating the row."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    with patch.object(GrblSerialDriver, "supports_probing", False):
        page._update_devices([_discovered_device()])
    row = page._device_rows["GrblSerialDriver:/dev/ttyUSB0"]
    assert row.select_button.get_label() == "Select"

    selected = {}
    page.device_selected.connect(
        lambda sender, **kw: selected.update(kw), weak=False
    )
    row.select_button.emit("clicked")
    assert selected["device"].key == "GrblSerialDriver:/dev/ttyUSB0"


@pytest.mark.ui
def test_discover_page_next_is_configure_manually(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    assert wizard.stack.get_visible_child_name() == "discover"
    assert wizard.next_btn.get_label() == "Configure Manually"
    assert wizard.next_btn.get_visible()
    assert wizard.next_btn.get_sensitive()


@pytest.mark.ui
def test_discover_configure_manually_goes_to_profile(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    wizard.next_btn.emit("clicked")
    assert wizard.stack.get_visible_child_name() == "profile"


@pytest.mark.ui
def test_device_selected_with_match_skips_to_probe(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    matched = DeviceProfile(
        meta=DeviceMeta(name="Frob One", vendor="Frobnicate"),
        machine_config=MachineConfig(driver="GrblSerialDriver"),
        dialect_config={},
    )
    device = _discovered_device()
    # Keep the auto-started probe on the next step inert.
    with (
        patch("rayforge.ui_gtk.machine.wizard_pages.probe_page.task_mgr"),
        patch.object(
            type(ui_context_initializer.device_profile_mgr),
            "match_device",
            return_value=matched,
        ),
    ):
        wizard._on_device_selected(page, device=device)

    assert wizard.stack.get_visible_child_name() == "probe"
    assert wizard.profile.name == "Frob One"
    mc = wizard.profile.machine_config
    assert mc.driver == "GrblSerialDriver"
    assert mc.driver_args == {"port": "/dev/ttyUSB0", "baudrate": 115200}
    assert {"profile", "controller", "connect"} <= wizard._skipped_steps_set


@pytest.mark.ui
def test_probe_result_adopts_matched_profile(ui_context_initializer):
    """The probe's build info names the device, so the wizard re-runs
    profile matching: a unique match adopts the curated profile and
    reroutes to the known-machine flow."""
    wizard = _make_wizard(ui_context_initializer)
    discover_page = wizard._get_page("discover")
    assert isinstance(discover_page, DiscoverPage)

    with patch("rayforge.ui_gtk.machine.wizard_pages.probe_page.task_mgr"):
        wizard._on_device_selected(
            discover_page, device=_discovered_device(identity=DeviceIdentity())
        )
    assert wizard.stack.get_visible_child_name() == "probe"
    assert wizard._source is None

    # The probe returns a profile built from $I/$$ answers.
    probed = DeviceProfile(
        meta=DeviceMeta(name="Frob One"),
        machine_config=MachineConfig(
            driver="GrblSerialDriver",
            driver_args={"port": "/dev/ttyUSB0", "baudrate": 115200},
            driver_config={"rx_buffer_size": 31},
            axis_extents=(120.0, 120.0),
        ),
        dialect_config={},
    )
    curated = DeviceProfile(
        meta=DeviceMeta(name="Frob One", vendor="Frobnicate"),
        machine_config=MachineConfig(
            driver="GrblSerialDriver",
            axis_extents=(150.0, 150.0),
        ),
        dialect_config={},
    )
    with patch.object(
        type(ui_context_initializer.device_profile_mgr),
        "match_device",
        return_value=curated,
    ):
        wizard._on_probe_succeeded(
            None, profile=probed, warnings=["laser mode off"]
        )

    assert wizard._source is not None
    assert wizard._source["kind"] == "profile"
    assert wizard.profile.name == "Frob One"
    mc = wizard.profile.machine_config
    assert mc.driver_args == {"port": "/dev/ttyUSB0", "baudrate": 115200}
    # Live connection facts survive; curated specs are trusted over
    # the probe's readings.
    assert mc.driver_config == {"rx_buffer_size": 31}
    assert mc.axis_extents == (150.0, 150.0)

    # Known-machine flow: everything through camera is skipped; the
    # profile is the source of truth for optional hardware too.
    assert wizard._next_step_after("probe") == "review"
    assert {
        "ai_provider",
        "ai_lookup",
        "hardware",
        "head",
        "rotary",
        "camera",
    } <= wizard._skipped_steps_set


@pytest.mark.ui
def test_device_selected_without_match_probes_device(ui_context_initializer):
    """No profile match, but a probe-capable driver: ask the device
    itself instead of making the user pick from the catalog."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    # No tokens → no profile can match; the probe auto-starts on
    # entry, so keep its task manager inert.
    with patch("rayforge.ui_gtk.machine.wizard_pages.probe_page.task_mgr"):
        wizard._on_device_selected(
            page, device=_discovered_device(identity=DeviceIdentity())
        )

    assert wizard.stack.get_visible_child_name() == "probe"
    mc = wizard.profile.machine_config
    assert mc.driver == "GrblSerialDriver"
    assert mc.driver_args == {"port": "/dev/ttyUSB0", "baudrate": 115200}
    assert wizard.aux_state.get("discovered") is not None
    assert {"controller", "connect"} <= wizard._skipped_steps_set


@pytest.mark.ui
def test_device_selected_without_match_shows_profile(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    # A driver that cannot probe falls back to the profile picker.
    with patch.object(GrblSerialDriver, "supports_probing", False):
        wizard._on_device_selected(
            page, device=_discovered_device(identity=DeviceIdentity())
        )

    assert wizard.stack.get_visible_child_name() == "profile"
    mc = wizard.profile.machine_config
    assert mc.driver == "GrblSerialDriver"
    assert mc.driver_args == {"port": "/dev/ttyUSB0", "baudrate": 115200}

    profile_page = wizard._get_page("profile")
    assert isinstance(profile_page, ProfilePage)
    assert profile_page.hint_row.get_visible()

    # Picking "Device Not Listed" keeps the discovered driver and
    # collected data; with complete connection args the remaining
    # known-data steps are skipped and the unknown-machine flow
    # (AI lookup) begins.
    wizard._on_profile_source_selected(
        profile_page, kind="other", profile=None
    )
    assert wizard.stack.get_visible_child_name() == "ai_provider"
    assert wizard.profile.machine_config.driver == "GrblSerialDriver"
    assert wizard.profile.machine_config.driver_args == {
        "port": "/dev/ttyUSB0",
        "baudrate": 115200,
    }


def _octoprint_device():
    """A device discovered over mDNS, as network discovery yields it."""
    return _discovered_device(
        driver_name="OctoPrintDriver",
        params={"host": "192.168.1.42", "port": 80},
        label="OctoPrint",
        detail="192.168.1.42:80 (octopi.local)",
        identity=DeviceIdentity(
            firmware="octoprint", banner="OctoPrint on octopi"
        ),
    )


@pytest.mark.ui
def test_discover_page_lists_network_device(ui_context_initializer):
    """mDNS-found devices appear like serial ones, are never held on
    a serial port, and vanish again when the network loses them."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    page._update_devices([_octoprint_device()])
    key = "OctoPrintDriver:192.168.1.42:80"
    assert list(page._device_rows) == [key]
    assert page._held_ports == set()
    row = page._device_rows[key]
    assert row.get_title() == "OctoPrint"
    subtitle = row.get_subtitle()
    assert subtitle is not None
    assert "192.168.1.42:80" in subtitle

    # No serial port to hold: an empty rescan drops the row.
    page._update_devices([])
    assert list(page._device_rows) == []


@pytest.mark.ui
def test_octoprint_device_selected_goes_to_connect(ui_context_initializer):
    """Selecting a discovered OctoPrint server keeps the resolved
    address and asks for the missing API key on the connect page."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    with patch.object(
        type(ui_context_initializer.device_profile_mgr),
        "match_device",
        return_value=None,
    ):
        wizard._on_device_selected(page, device=_octoprint_device())

    # host/port alone don't complete the connection args (no API
    # key yet), so the profile picker comes first.
    assert wizard.stack.get_visible_child_name() == "profile"
    mc = wizard.profile.machine_config
    assert mc.driver == "OctoPrintDriver"
    assert mc.driver_args == {"host": "192.168.1.42", "port": 80}

    profile_page = wizard._get_page("profile")
    assert isinstance(profile_page, ProfilePage)
    wizard._on_profile_source_selected(
        profile_page, kind="other", profile=None
    )

    assert wizard.stack.get_visible_child_name() == "connect"
    connect_page = wizard._get_page("connect")
    assert isinstance(connect_page, ConnectionPage)
    values = connect_page.connect_widget.get_values()
    assert values.get("host") == "192.168.1.42"
    assert values.get("port") == 80
    # The API key is still missing: not ready to continue.
    assert connect_page.ready is False


def _probed_profile(name="Mystery CNC", **machine_overrides):
    return DeviceProfile(
        meta=DeviceMeta(name=name),
        machine_config=MachineConfig(
            driver="GrblSerialDriver", **machine_overrides
        ),
        dialect_config={},
    )


@pytest.mark.ui
def test_device_selected_with_probe_data_and_match_completes_flow(
    ui_context_initializer,
):
    """Probe data collected on the discover page + a unique profile
    match: everything through the camera step is skipped and the
    user lands directly on the review."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    probed = _probed_profile(
        "Frob One",
        driver_config={"rx_buffer_size": 31},
        axis_extents=(120.0, 120.0),
    )
    curated = DeviceProfile(
        meta=DeviceMeta(name="Frob One", vendor="Frobnicate"),
        machine_config=MachineConfig(
            driver="GrblSerialDriver", axis_extents=(150.0, 150.0)
        ),
        dialect_config={},
    )
    device = _discovered_device(
        identity=DeviceIdentity(), probe_profile=probed
    )
    with patch.object(
        type(ui_context_initializer.device_profile_mgr),
        "match_device",
        return_value=curated,
    ):
        wizard._on_device_selected(page, device=device)

    assert wizard.stack.get_visible_child_name() == "review"
    assert wizard.profile.name == "Frob One"
    mc = wizard.profile.machine_config
    # Live connection facts survive; curated specs are trusted.
    assert mc.driver_args == {"port": "/dev/ttyUSB0", "baudrate": 115200}
    assert mc.driver_config == {"rx_buffer_size": 31}
    assert mc.axis_extents == (150.0, 150.0)
    assert {
        "profile",
        "controller",
        "connect",
        "probe",
        "ai_provider",
        "ai_lookup",
        "hardware",
        "head",
        "rotary",
        "camera",
    } <= wizard._skipped_steps_set


@pytest.mark.ui
def test_device_selected_with_probe_data_without_match_filters_profiles(
    ui_context_initializer,
):
    """Probe data without a match: the profile picker comes up (not
    the probe page), the probed specs are already in place, and
    picking a profile completes the known-machine flow."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    probed = _probed_profile(axis_extents=(120.0, 120.0))
    device = _discovered_device(
        identity=DeviceIdentity(), probe_profile=probed
    )
    wizard._on_device_selected(page, device=device)

    assert wizard.stack.get_visible_child_name() == "profile"
    assert {"controller", "connect", "probe"} <= wizard._skipped_steps_set
    assert wizard.profile.machine_config.axis_extents == (120.0, 120.0)

    profile_page = wizard._get_page("profile")
    assert isinstance(profile_page, ProfilePage)

    curated = DeviceProfile(
        meta=DeviceMeta(name="Frob One", vendor="Frobnicate"),
        machine_config=MachineConfig(
            driver="GrblSerialDriver", axis_extents=(150.0, 150.0)
        ),
        dialect_config={},
    )
    wizard._on_profile_source_selected(
        profile_page, kind="profile", profile=curated
    )
    assert wizard.stack.get_visible_child_name() == "review"
    assert wizard.profile.machine_config.axis_extents == (150.0, 150.0)
    assert wizard.profile.machine_config.driver_args == {
        "port": "/dev/ttyUSB0",
        "baudrate": 115200,
    }


@pytest.mark.ui
def test_discover_page_probes_and_enriches_found_device(
    ui_context_initializer,
):
    """A found device is probed automatically; the row then shows
    the machine's own name and work area."""
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    device = _discovered_device()
    with (
        patch(
            "rayforge.ui_gtk.machine.wizard_pages.discover_page.get_context",
            return_value=ui_context_initializer,
        ),
        patch(
            "rayforge.ui_gtk.machine.wizard_pages.discover_page.task_mgr"
        ) as tm,
    ):
        page._update_devices([device])
        assert tm.add_coroutine.called
        assert page._held_ports == {"/dev/ttyUSB0"}

        probed = _probed_profile("Sculpfun iCube", axis_extents=(120.0, 120.0))
        task = MagicMock()
        task.result.return_value = (probed, [])
        tm.schedule_on_main_thread.side_effect = lambda fn: fn()
        page._on_probe_done(task, device)

    row = page._device_rows["GrblSerialDriver:/dev/ttyUSB0"]
    assert row.get_title() == "Sculpfun iCube"
    assert row.device.probe_profile is probed
    subtitle = row.get_subtitle()
    assert subtitle is not None
    assert "/dev/ttyUSB0 at 115200 baud" in subtitle
    assert "120" in subtitle

    # The enriched device is what selection emits.
    selected = {}
    page.device_selected.connect(
        lambda sender, **kw: selected.update(kw), weak=False
    )
    page._on_row_activated(row)
    assert selected["device"].probe_profile is probed


@pytest.mark.ui
def test_discover_page_rescan_excludes_and_prunes(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    page = wizard._get_page("discover")
    assert isinstance(page, DiscoverPage)

    with patch.object(GrblSerialDriver, "supports_probing", False):
        page._update_devices([_discovered_device()])
    assert page._held_ports == {"/dev/ttyUSB0"}

    # Rescans keep away from ports whose devices are already held.
    captured = {}

    async def fake_find(driver_classes=None, exclude_ports=None, **kwargs):
        captured["exclude_ports"] = set(exclude_ports or ())

    with (
        patch(
            "rayforge.ui_gtk.machine.wizard_pages.discover_page"
            ".find_all_devices",
            fake_find,
        ),
        patch(
            "rayforge.ui_gtk.machine.wizard_pages.discover_page.task_mgr"
        ) as tm,
    ):
        page._start_scan()
        coro = tm.add_coroutine.call_args.args[0]
        asyncio.run(coro(None))
    assert captured["exclude_ports"] == {"/dev/ttyUSB0"}

    # A device that is unplugged is pruned and its port freed.
    with patch(
        "rayforge.ui_gtk.machine.wizard_pages.discover_page.SerialTransport"
    ) as transport:
        transport.list_port_info.return_value = []
        page._prune_unplugged_ports()
    assert page._device_rows == {}
    assert page._held_ports == set()


@pytest.mark.ui
def test_profile_page_suggests_matching_vendor(ui_context_initializer):
    """With a pending discovered device, the profile list narrows to
    profiles of the same vendor and controller; searching broadens
    it back to the full catalog."""
    from rayforge.machine.driver import normalize_tokens

    wizard = _make_wizard(ui_context_initializer)
    profile_page = wizard._get_page("profile")
    assert isinstance(profile_page, ProfilePage)

    frob = DeviceProfile(
        meta=DeviceMeta(name="Frob One", vendor="Frobnicate"),
        machine_config=MachineConfig(driver="GrblSerialDriver"),
        dialect_config={},
    )
    other = DeviceProfile(
        meta=DeviceMeta(name="Gadget Two", vendor="Gadgetco"),
        machine_config=MachineConfig(driver="GrblSerialDriver"),
        dialect_config={},
    )
    device = _discovered_device(
        identity=DeviceIdentity(tokens=normalize_tokens("Frobnicate laser"))
    )
    wizard.aux_state = {"discovered": device}

    mgr = ui_context_initializer.device_profile_mgr
    with (
        patch(
            "rayforge.ui_gtk.machine.wizard_pages.profile_page.get_context",
            return_value=ui_context_initializer,
        ),
        patch.object(mgr, "get_all", return_value=[frob, other]),
    ):
        profile_page.enter(wizard.profile)
        titles = _visible_profile_titles(profile_page)
        assert titles == ["Frob One"]

        # Searching broadens the list back to the full catalog.
        # (search-changed is delay-debounced by GTK, so filter
        # explicitly here.)
        profile_page.search_entry.set_text("gadget")
        profile_page._filter_and_populate_list()
        titles = _visible_profile_titles(profile_page)
        assert titles == ["Gadget Two"]


def _visible_profile_titles(page: ProfilePage) -> list[str]:
    titles = []
    index = 0
    while (row := page.list_box.get_row_at_index(index)) is not None:
        titles.append(row.get_title())  # type: ignore[attr-defined]
        index += 1
    return titles
