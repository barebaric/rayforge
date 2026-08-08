"""UI tests for the Unified Machine Configuration Wizard."""

from unittest.mock import MagicMock, patch

import pytest
from gi.repository import Adw

from rayforge.machine.device.profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)
from rayforge.machine.models.machine import Origin
from rayforge.ui_gtk.machine.unified_wizard import UnifiedWizard
from rayforge.ui_gtk.machine.wizard_pages.ai_lookup_page import AILookupPage
from rayforge.ui_gtk.machine.wizard_pages.camera_page import CameraPage
from rayforge.ui_gtk.machine.wizard_pages.connection_page import ConnectionPage
from rayforge.ui_gtk.machine.wizard_pages.controller_page import ControllerPage
from rayforge.ui_gtk.machine.wizard_pages.probe_page import ProbePage
from rayforge.ui_gtk.machine.wizard_pages.provider_page import AIProviderPage
from rayforge.ui_gtk.machine.wizard_pages.review_page import ReviewPage

_CAMERA_PATCH_TARGET = (
    "rayforge.ui_gtk.machine.wizard_pages.camera_page.get_sorted_by_id_paths"
)


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
def test_wizard_starts_on_profile_page(ui_context_initializer):
    wizard = _make_wizard(ui_context_initializer)
    assert wizard.stack.get_visible_child_name() == "profile"


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
def test_known_profile_skips_ai_hw_head_pages(ui_context_initializer):
    """A picked profile carries trusted specs, so after Connection the
    wizard skips probe, AI, hardware and head — landing on Rotary.
    Imports are not fully reliable, so they keep hardware/head (but
    still skip the AI pages)."""
    wizard = _make_wizard(ui_context_initializer)
    wizard._on_profile_source_selected(
        None, kind="profile", profile=_profile(driver="RuidaDriver")
    )
    assert wizard._next_step_after("connect") == "rotary"
    assert {
        "controller",
        "probe",
        "ai_provider",
        "ai_lookup",
        "hardware",
        "head",
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
    """Step 1 advances only via explicit source selection, so the
    generic Next button must not appear next to "Other / Unknown
    Device…" — otherwise the two look like duplicate primary actions."""
    wizard = _make_wizard(ui_context_initializer)
    assert not wizard.next_btn.get_visible()


@pytest.mark.ui
def test_back_visible_and_returns_to_profile_after_source(
    ui_context_initializer,
):
    wizard = _make_wizard(ui_context_initializer)
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
        "profile",
        "controller",
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
