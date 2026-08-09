"""The Unified Machine Configuration Wizard.

Orchestrates the multi-step machine setup flow. Holds a
working ``DeviceProfile`` in memory and routes between pages
dynamically based on the user's choices (known profile vs unknown,
probe-capable controller vs none, AI-lookup available vs not, etc.).

Emits ``machine_created`` with the resulting live ``Machine`` once the
user completes the final review step.

Replaces the legacy ``MachineProfileSelectorDialog`` + ``ConfigWizard``
pair.
"""

import logging
from gettext import gettext as _
from typing import Any

from blinker import Signal
from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ...camera.models.camera import Camera
from ...camera.v4l import display_name
from ...context import get_context
from ...core.ai.spec_lookup import is_ai_configured
from ...machine.device.profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)
from ...machine.driver import get_driver_cls
from ...machine.driver.dummy import NoDeviceDriver
from ..camera.wizard.wizard import CameraWizard
from ..shared.patched_dialog_window import PatchedDialogWindow
from .wizard_pages import WizardPage, empty_profile
from .wizard_pages.ai_lookup_page import AILookupPage
from .wizard_pages.camera_page import CameraPage
from .wizard_pages.connection_page import ConnectionPage
from .wizard_pages.controller_page import ControllerPage
from .wizard_pages.hardware_page import HardwarePage
from .wizard_pages.head_page import HeadPage
from .wizard_pages.probe_page import ProbePage
from .wizard_pages.profile_page import ProfilePage
from .wizard_pages.provider_page import AIProviderPage
from .wizard_pages.review_page import ReviewPage
from .wizard_pages.rotary_page import RotaryPage

logger = logging.getLogger(__name__)


# Ordered list of step names the wizard knows about. Adaptive routing
# may skip individual entries based on the user's choices.
_STEP_ORDER: list[str] = [
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


class UnifiedWizard(PatchedDialogWindow):
    """The unified add-machine wizard dialog.

    A ``PatchedDialogWindow`` subclass that hosts one
    :class:`WizardPage` at a time, drives a footer with Back / Next /
    Create / Cancel buttons, and manages the in-memory working
    ``DeviceProfile``. The orchestrator decides — based on the current
    page's outcome — which page to show next.
    """

    def __init__(self, **kwargs):
        self.profile_created = Signal()
        super().__init__(
            transient_for=kwargs.pop("transient_for", None),
            modal=True,
            default_width=760,
            default_height=640,
            title=_("Add a Machine"),
            **kwargs,
        )

        self.profile: DeviceProfile = empty_profile()
        # Aux state for pages that need to carry session-only fields
        # not part of DeviceProfile (e.g. axis reversals applied at
        # machine-creation time).
        self.aux_state: dict[str, Any] = {}
        # Set when the user picks a known profile or import on Step 1;
        # None for "Other / unknown machine". Used to skip the AI
        # lookup steps when the specs are already in the profile.
        self._source: dict[str, Any] | None = None

        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        self.header = Adw.HeaderBar()
        content.append(self.header)

        self._main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        content.append(self._main_box)

        self.stack = Gtk.Stack()
        self.stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self.stack.set_vexpand(True)
        self.stack.connect("notify::visible-child", self._on_page_changed)
        self._main_box.append(self.stack)

        # Lazily-built pages keyed by step name. We construct each
        # page only when the user visits it so that pages that touch
        # hardware (e.g. probe) only initialize when relevant.
        self._pages: dict[str, WizardPage] = {}

        # History stack — supports the Back button.
        self._history: list[str] = []

        # Steps we deliberately won't re-enter when the user presses
        # Back — populated as the user proceeds (e.g. "controller" gets
        # added when the user picks a known profile or import).
        self._skipped_steps_set: set = set()

        self._build_buttons(self._main_box)

        # Initial state: profile page is step 1.
        self._navigate_to("profile", record_history=False)

    # ----- public API ----------------------------------------------------

    def show_error(self, heading: str, body: str) -> None:
        """Convenience: surface a transient error to the user."""
        toast = Adw.Toast.new(f"{heading}: {body}" if body else heading)
        toast.set_timeout(5)
        self.toast_overlay.add_toast(toast)

    # ----- footer -------------------------------------------------------

    def _build_buttons(self, main_box: Gtk.Box) -> None:
        self._button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_top=12,
        )
        # Like the camera calibration wizard, the button bar lives
        # inside the main box, directly under the stack, so it shares
        # the content margins.
        main_box.append(self._button_box)

        self.back_btn = Gtk.Button(label=_("Back"))
        self.back_btn.add_css_class("flat")
        self.back_btn.connect("clicked", self._on_back_clicked)
        self.back_btn.set_visible(False)
        self._button_box.append(self.back_btn)

        self.cancel_btn = Gtk.Button(label=_("Cancel"))
        self.cancel_btn.add_css_class("flat")
        self.cancel_btn.connect("clicked", lambda _: self.close())
        self._button_box.append(self.cancel_btn)

        self.skip_btn = Gtk.Button(label=_("Skip"))
        self.skip_btn.add_css_class("flat")
        self.skip_btn.connect("clicked", self._on_skip_clicked)
        self.skip_btn.set_visible(False)
        self._button_box.append(self.skip_btn)

        # Slot for page-specific action buttons (e.g. "Probe Now",
        # "Look Up Specs"). Repopulated per page in _update_footer().
        self._action_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
        )
        self._button_box.append(self._action_box)
        self._footer_action_buttons: list[Gtk.Button] = []

        self.next_btn = Gtk.Button(label=_("Next"))
        self.next_btn.add_css_class("suggested-action")
        self.next_btn.connect("clicked", self._on_next_clicked)
        self._button_box.append(self.next_btn)

        self.create_btn = Gtk.Button(label=_("Create Machine"))
        self.create_btn.add_css_class("suggested-action")
        self.create_btn.connect("clicked", self._on_create_clicked)
        self.create_btn.set_visible(False)
        self._button_box.append(self.create_btn)

    # ----- lazy page access ---------------------------------------------

    def _get_page(self, name: str) -> WizardPage | None:
        # Step 1 traverses the wizard in declared order — only "next",
        # "back", "skip", and the adaptive router call this.
        if name == "profile":
            cls = ProfilePage
        elif name == "controller":
            cls = ControllerPage
        elif name == "connect":
            cls = ConnectionPage
        elif name == "probe":
            cls = ProbePage
        elif name == "ai_provider":
            cls = AIProviderPage
        elif name == "ai_lookup":
            cls = AILookupPage
        elif name == "hardware":
            cls = HardwarePage
        elif name == "head":
            cls = HeadPage
        elif name == "rotary":
            cls = RotaryPage
        elif name == "camera":
            cls = CameraPage
        elif name == "review":
            cls = ReviewPage
        else:
            return None

        if name in self._pages:
            return self._pages[name]

        try:
            page = cls(self)
            page.ready_changed.connect(self._on_ready_changed)
            self._pages[name] = page
            self.stack.add_named(page, name)

            # Wire page-specific signals
            self._wire_page_signals(page, name)

            return page
        except Exception:
            logger.exception("Failed to build page %s", name)
            return None

    def _wire_page_signals(self, page: WizardPage, name: str) -> None:
        if isinstance(page, ProfilePage):
            page.source_selected.connect(self._on_profile_source_selected)
        elif isinstance(page, ControllerPage):
            page.controller_selected.connect(self._on_controller_selected)
        elif isinstance(page, ProbePage):
            page.probe_succeeded.connect(self._on_probe_succeeded)

    # ----- navigation ----------------------------------------------------

    def _navigate_to(self, name: str, *, record_history: bool = True) -> None:
        if record_history:
            previous = self.stack.get_visible_child_name()
            if previous is not None:
                self._history.append(previous)

        page = self._get_page(name)
        if page is None:
            logger.error("Unknown wizard step: %s", name)
            return

        page.enter(self.profile)
        self.stack.set_visible_child_name(name)
        self._update_footer(name, page)

    def _on_page_changed(self, _stack, _param) -> None:
        name = self.stack.get_visible_child_name()
        if not name:
            return
        self._update_footer(name, self._get_page(name))

    def _update_footer(self, name: str, page: WizardPage | None) -> None:
        if page is None:
            return

        # Repopulate the page-action slot so every interactive
        # affordance lives on the button bar.
        for btn in self._footer_action_buttons:
            self._action_box.remove(btn)
        self._footer_action_buttons.clear()
        for btn in page.footer_buttons():
            self._action_box.append(btn)
            self._footer_action_buttons.append(btn)

        # Header title reflects the current step.
        self.set_title(page.title or _("Add a Machine"))

        # Back button visible when there's history.
        self.back_btn.set_visible(bool(self._history))

        # Skip button is only meaningful on optional steps where it is
        # semantically distinct from Next: the AI provider page (skip =
        # decline AI), rotary, and camera. The probe and AI lookup pages
        # are always-ready, so Skip there would duplicate Next.
        if name in ("ai_provider", "rotary", "camera"):
            self.skip_btn.set_visible(True)
        else:
            self.skip_btn.set_visible(False)

        # Next vs Create: Create replaces Next on the final step
        # ("review"). Pages that opt out of the generic Next (e.g.
        # Step 1, which advances via explicit source selection) hide
        # it entirely so no dead buttons sit on the footer.
        self.next_btn.set_visible(name != "review" and page.next_on_footer)
        self.create_btn.set_visible(name == "review")

        # Next sensitivity follows page readiness.
        self.next_btn.set_sensitive(bool(page.ready))
        self.create_btn.set_sensitive(True)

    def _on_ready_changed(self, page: WizardPage, **kwargs) -> None:
        if page is not self.stack.get_visible_child():
            return
        name = self.stack.get_visible_child_name()
        if name == "review":
            self.create_btn.set_sensitive(True)
        else:
            self.next_btn.set_sensitive(page.ready)

    # ----- footer button handlers --------------------------------------

    def _on_next_clicked(self, _btn: Gtk.Button) -> None:
        name = self.stack.get_visible_child_name()
        if name is None:
            return
        page = self._get_page(name)
        if page is None or not page.apply_to_profile(self.profile):
            return
        if name == "camera":
            self._after_camera_next()
            return
        next_step = self._next_step_after(name)
        if next_step is None:
            return
        self._navigate_to(next_step)

    def _on_skip_clicked(self, _btn: Gtk.Button) -> None:
        name = self.stack.get_visible_child_name()
        if name is None:
            return
        next_step = self._next_step_after(name)
        if next_step is None:
            return
        self._navigate_to(next_step)

    def _on_back_clicked(self, _btn: Gtk.Button | None) -> None:
        if not self._history:
            return
        prev = self._history.pop()
        # Re-enter previous page; historical loads don't push the
        # still-current page onto history again.
        page = self._get_page(prev)
        if page is None:
            return
        page.enter(self.profile)
        self.stack.set_visible_child_name(prev)
        self.back_btn.set_visible(bool(self._history))

    def _on_create_clicked(self, _btn: Gtk.Button) -> None:
        page = self._get_page("review")
        if page is not None and not page.apply_to_profile(self.profile):
            return
        try:
            machine = self._materialize_machine()
        except Exception as exc:
            logger.exception("Failed to create machine")
            self.show_error(_("Could not create machine"), str(exc))
            return
        self.profile_created.send(self, profile=self.profile, machine=machine)
        self.close()

    def _materialize_machine(self):
        context = get_context()
        machine = self.profile.create_machine(context)

        # Apply session-only aux_state on the live machine.
        reverse = self.aux_state.get("reverse", {})
        if reverse.get("x"):
            machine.set_reverse_x_axis(True)
        if reverse.get("y"):
            machine.set_reverse_y_axis(True)
        if reverse.get("z"):
            machine.set_reverse_z_axis(True)
        return machine

    # ----- adaptive routing --------------------------------------------

    def _source_kind(self) -> str | None:
        return self._source.get("kind") if self._source else None

    def _ai_entry_step(self) -> str:
        """Where the wizard enters the AI flow after probing/connection.

        A known profile or import already carries the machine specs,
        so the AI provider / lookup steps are skipped entirely. A known
        *profile* also trusts the work-area and head specs, so the
        hardware and head steps are skipped too (the user still adds
        rotary modules / cameras). Imports are not 100% reliable, so
        the user is walked through hardware and head with prefilled
        values they can correct. For "Other / unknown machine", the
        user is first asked on the provider page when none is
        configured; otherwise the lookup page comes up directly.
        """
        kind = self._source_kind()
        if kind == "profile":
            self._skipped_steps_set.update(
                {"ai_provider", "ai_lookup", "hardware", "head"}
            )
            return "rotary"
        if kind == "import":
            self._skipped_steps_set.update({"ai_provider", "ai_lookup"})
            return "hardware"
        return "ai_lookup" if is_ai_configured() else "ai_provider"

    def _next_step_after(self, name: str) -> str | None:
        """Decides the next step using the adaptive routing rules."""
        mc = self.profile.machine_config

        if name == "profile":
            # Routing is set by source_selected signal. If we got here
            # via the plain Next button on the profile page (shouldn't
            # normally happen since the page emits source_selected),
            # fall through to controller.
            return "controller"

        if name == "controller":
            # `None` controller skips Steps 3 & 4 entirely.
            if not mc.driver:
                self._skipped_steps_set.update({"connect", "probe"})
                return self._ai_entry_step()
            return "connect"

        if name == "connect":
            # A known profile already carries trusted specs, so skip
            # the auto-discovery probe entirely. Imports are not fully
            # reliable, so the user may still probe (when the driver
            # supports it) to verify/correct the imported values.
            if self._source_kind() == "profile":
                self._skipped_steps_set.update({"probe"})
                return self._ai_entry_step()
            # Probe page only if driver supports probing.
            driver_cls = None
            if mc.driver:
                driver_cls = get_driver_cls(mc.driver)
                # The NoDeviceDriver fallback means "None — G-code
                # export only"; there is nothing to probe with.
                if driver_cls is NoDeviceDriver:
                    driver_cls = None
            if driver_cls is not None and driver_cls.supports_probing:
                return "probe"
            self._skipped_steps_set.update({"probe"})
            return self._ai_entry_step()

        if name == "probe":
            return self._ai_entry_step()

        if name == "ai_provider":
            # Next means the provider was configured; Skip means the
            # user declined AI, so skip the lookup page entirely.
            return "ai_lookup" if is_ai_configured() else "hardware"

        if name == "ai_lookup":
            return "hardware"

        if name == "hardware":
            return "head"

        if name == "head":
            return "rotary"

        if name == "rotary":
            return "camera"

        if name == "camera":
            return "review"

        return None

    # ----- page-specific signal handlers --------------------------------

    def _on_controller_selected(self, sender, *, driver: str | None) -> None:
        """Step 2: the user picked a controller tile — advance at once."""
        self.profile.machine_config.driver = driver
        next_step = self._next_step_after("controller")
        if next_step is None:
            return
        self._navigate_to(next_step)

    def _on_profile_source_selected(
        self, sender, *, kind: str, profile: DeviceProfile | None
    ) -> None:
        """Step 1: the user picked a starting point."""
        if kind == "other":
            # Start fresh; the controller page takes over.
            self.profile = empty_profile()
            self.aux_state = {}
            self._source = None
        elif kind in ("profile", "import") and profile is not None:
            # Adopt the picked profile's data as our working state; we
            # still require Step 3 (Connection) per design decision #5.
            self.profile = self._clone_profile(profile)
            self.aux_state = {}
            # Stash chosen source for later sanity feedback
            self._source = {"kind": kind, "profile": profile}
        else:
            self.profile = empty_profile()
            self.aux_state = {}
            self._source = None

        # Step 1 → Step 3 for known/import, Step 1 → Step 2 for "Other".
        # Navigate with history recording so "profile" stays on the
        # stack: the user can press Back to change their source choice.
        target: str
        if kind == "other":
            target = "controller"
        else:
            # Jump straight to connection (the picked profile fixes the
            # controller); Back returns to the profile picker.
            self._skipped_steps_set.add("controller")
            target = "connect"
        self._navigate_to(target)

    def _on_probe_succeeded(
        self, sender, *, profile: DeviceProfile, warnings: list[str]
    ) -> None:
        """Step 4: the probe merged values into a working profile."""
        # Merge probed machine_config fields into our working profile.
        # Probe returns a full DeviceProfile; we overlay every
        # non-None field of its machine_config onto our profile.
        src = profile.machine_config
        dst = self.profile.machine_config
        for field_name in (
            "driver_args",
            "driver_config",
            "axis_extents",
            "origin",
            "max_travel_speed",
            "max_cut_speed",
            "acceleration",
            "single_axis_homing_enabled",
            "home_on_start",
            "heads",
            "unit_system",
        ):
            value = getattr(src, field_name, None)
            if value is not None:
                setattr(dst, field_name, value)
        for text in warnings:
            logger.info("Probing warning: %s", text)

    # ----- camera workflow ----------------------------------------------

    def _after_camera_next(self) -> None:
        """Step 10: when cameras were enabled, route into the camera
        workflow (lens calibration) for the first enabled device before
        landing on Review."""
        page = self._get_page("camera")
        enabled = (
            page.selected_device_ids() if isinstance(page, CameraPage) else []
        )
        self._navigate_to("review")
        if not enabled:
            return
        if not self._launch_camera_workflow(enabled[0]):
            self.show_error(
                _("Camera setup unavailable"),
                _(
                    "Calibrate this camera later from the machine "
                    "settings page."
                ),
            )

    def _launch_camera_workflow(self, device_id: str) -> bool:
        """Present the camera workflow dialog for *device_id*.

        Returns False when the camera backend is unavailable so the
        caller can fall back gracefully.
        """
        try:
            config = Camera(name=display_name(device_id), device_id=device_id)
            controller = CameraController(config)
            dialog = CameraWizard(self, controller)
        except Exception:
            logger.exception("Failed to start camera workflow")
            return False
        dialog.present()
        return True

    # ----- helpers -------------------------------------------------------

    def _clone_profile(self, src: DeviceProfile) -> DeviceProfile:
        """Adopt an existing profile's data into our working copy."""
        return DeviceProfile(
            meta=_clone_meta(src),
            machine_config=_clone_machine_config(src.machine_config),
            dialect_config=dict(src.dialect_config),
            source_dir=src.source_dir,
        )

    def close(self):
        super().close()

    def do_close_request(self, *args) -> bool:
        return False


def _clone_meta(src) -> Any:
    return DeviceMeta(
        name=src.name,
        vendor=src.meta.vendor,
        model=src.meta.model,
        description=src.meta.description,
        api_version=src.meta.api_version,
    )


def _clone_machine_config(src) -> Any:
    """Deep-copy a MachineConfig into a new mutable instance."""
    out = MachineConfig()
    for attr in (
        "driver",
        "driver_args",
        "driver_config",
        "gcode_precision",
        "supports_arcs",
        "supports_curves",
        "axis_extents",
        "work_margins",
        "soft_limits",
        "origin",
        "workspace_orientation",
        "max_travel_speed",
        "max_cut_speed",
        "home_on_start",
        "acceleration",
        "single_axis_homing_enabled",
        "rotary_enabled_default",
        "unit_system",
        "heads",
        "capabilities",
        "hookmacros",
        "rotary_modules",
        "nogo_zones",
        "cameras",
    ):
        value = getattr(src, attr, None)
        if isinstance(value, dict):
            value = dict(value)
        elif isinstance(value, list) and (
            attr
            in (
                "heads",
                "hookmacros",
                "rotary_modules",
                "nogo_zones",
                "cameras",
                "capabilities",
                "driver_args",
            )
        ):
            # Lists of dicts and tuples need to be copied with the
            # inner structures preserved too.
            value = _copy_list_of_containers(value)
        setattr(out, attr, value)
    return out


def _copy_list_of_containers(value: list) -> list:
    """Shallow-but-not-too-shallow copy of a list of containers."""
    out = []
    for item in value:
        if isinstance(item, dict):
            out.append(dict(item))
        elif isinstance(item, list):
            out.append(_copy_list_of_containers(item))
        elif isinstance(item, tuple):
            out.append(tuple(item))
        else:
            out.append(item)
    return out


__all__ = ["_STEP_ORDER", "UnifiedWizard"]
