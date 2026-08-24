"""Per-step pages for the Unified Machine Configuration Wizard.

Each page is a self-contained widget responsible for a single stage of
the wizard (Step 1: profile pick, Step 2: controller choice, Step 3:
connection, etc.). Pages communicate back to the orchestrator
(:class:`~rayforge.ui_gtk.machine.unified_wizard.UnifiedWizard`) via
the ``ready`` flag and ``apply_to_profile`` mechanism rather than
emitting signals directly, so they remain testable in isolation.

Pages never mutate the working ``DeviceProfile`` themselves — they
hand back a partial dict / call :meth:`Step.apply_to_profile` so the
orchestrator keeps single-source-of-truth state and can re-route
later steps adaptively.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from blinker import Signal
from gi.repository import Adw, Gtk

from ....machine.device.profile import DeviceMeta, DeviceProfile, MachineConfig

if TYPE_CHECKING:
    from ..unified_wizard import UnifiedWizard


class WizardPage(Adw.Bin):
    """Base class for all wizard step pages.

    Subclasses MUST override :meth:`build_ui` to populate the page,
    and SHOULD override :meth:`enter` (called each time the orchestrator
    shows the page) and :meth:`apply_to_profile` (called before
    navigating away, e.g. on "Next").

    The base class wires up a vertical scrolled container with the
    standard 24px page margins so subclasses only need to append
    content to ``self.content``.
    """

    # Step number (1..10) for diagnostics / navigation. Subclasses
    # MUST set this.
    step_number: int = 0
    # Short human-readable title shown in the wizard header.
    title: str = ""
    # Optional subtitle / descriptive blurb.
    subtitle: str = ""
    # When False the wizard hides the "Next" button on this page: the
    # page advances only through its own explicit affordances (e.g.
    # Step 1, which fires source_selected on row activation).
    next_on_footer: bool = True
    # Optional override for the footer "Next" button label (e.g.
    # "Configure Manually" on the discover step).
    next_label: str | None = None

    def __init__(self, wizard: "UnifiedWizard", **kwargs):
        super().__init__(**kwargs)
        self.wizard = wizard
        self.ready: bool = False
        # Fires with ``ready=True/False`` whenever the page activates
        # or deactivates its Next/Create affordance. The orchestrator
        # connects to this to drive the footer button sensitivity.
        self.ready_changed = Signal()

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_vexpand(True)
        scrolled.set_hexpand(True)
        self.set_child(scrolled)

        self.content = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=24,
            margin_bottom=24,
            margin_start=24,
            margin_end=24,
        )
        scrolled.set_child(self.content)
        self.build_ui()

    # ----- subclass hooks -------------------------------------------------

    def build_ui(self) -> None:
        """Populate ``self.content`` with the page's widgets."""

    def enter(self, profile: DeviceProfile) -> None:
        """Called each time the orchestrator switches to this page.

        Subclasses can prefill widgets from *profile* (and metadata
        stored alongside it on the wizard, like ``wizard.aux_state``).
        Default implementation does nothing.
        """

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        """Push this page's UI values into *profile*.

        Returns ``True`` if the orchestrator may advance to the next
        step, ``False`` to abort the navigation (e.g. on a validation
        error). Subclasses should keep this idempotent and cheap; the
        orchestrator may call it redundantly to refresh state.
        Default implementation returns ``True``.
        """
        return True

    def footer_buttons(self) -> list[Gtk.Button]:
        """Action buttons this page wants in the wizard's footer bar.

        Pages construct their buttons once in :meth:`build_ui` and
        return the references here. The orchestrator places them in
        the footer's action slot each time the page is shown, keeping
        every interactive affordance on the button bar (the camera
        calibration wizard layout). Default returns no buttons.
        """
        return []

    # ----- helpers ---------------------------------------------------------

    def set_ready(self, ready: bool) -> None:
        """Signal that the page's "Next"/"Create" affordance can activate."""
        if ready == self.ready:
            return
        self.ready = ready
        self.ready_changed.send(self, ready=ready)


def empty_profile() -> DeviceProfile:
    """A blank working profile used as the wizard's initial state."""
    return DeviceProfile(
        meta=DeviceMeta(name=_("New Machine")),
        machine_config=MachineConfig(),
        dialect_config={},
        source_dir=None,
    )


def _makePreferencesGroup(
    title: str | None = None,
    description: str | None = None,
) -> Adw.PreferencesGroup:
    """Helper constructor that omits None titles entirely."""
    kwargs: dict[str, Any] = {}
    if title:
        kwargs["title"] = title
    if description:
        kwargs["description"] = description
    return Adw.PreferencesGroup(**kwargs)


__all__ = [
    "WizardPage",
    "_makePreferencesGroup",
    "empty_profile",
]
