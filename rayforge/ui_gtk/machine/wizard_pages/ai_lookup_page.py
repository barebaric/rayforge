"""Step 6 — AI spec lookup.

Queries the configured AI provider (default OpenAI-compatible
provider via :func:`~rayforge.core.ai.spec_lookup.lookup_machine_specs`)
by ``vendor + model`` and surfaces the returned fields as suggestion
rows the user can accept, edit, or reject:

* **AI not configured** → friendly "Configure AI in Settings…" link;
  the page auto-skips after the user dismisses the prompt.
* **AI returns a field** → shows as a switch row with the
  LLM-proposed value; each row starts switched on (accepted) and the
  user toggles off anything they don't want. Toggling on/off is the
  explicit accept/reject affordance.
* **Lookup errors / no parseable JSON** → degrades gracefully and
  shows an informational banner; the user moves on.

While a lookup runs, a thin pulsing progress bar (mirroring the AI
workpiece generator dialog) gives the user live feedback.
"""

from gettext import gettext as _
from typing import Any, Dict, List, Optional, Tuple

from blinker import Signal
from gi.repository import Adw, GLib, Gtk

from ....context import get_context
from ....core.ai.spec_lookup import is_ai_configured, lookup_machine_specs
from ....machine.device.profile import DeviceProfile
from ....machine.models.machine import Origin
from ....shared.tasker import task_mgr
from . import WizardPage, _makePreferencesGroup

# Fields the AI may return and the wizard knows how to merge. Each
# entry maps the JSON key to (target_machine_config_attr, label,
# kind). ``kind`` is one of:
#   "tuple2"    — pair of floats (axis_extents, spot_size_mm)
#   "int"       — integer scalar
#   "bool"      — boolean
#   "string"    — string scalar (origin)
#   "head_laser"  — head field, laser family
#   "head_spindle" — head field, spindle family
_FIELD_SPEC: List[Tuple[str, str, str]] = [
    ("axis_extents", _("Work area (X, Y)"), "tuple2_mm"),
    ("max_travel_speed", _("Max travel speed"), "int"),
    ("max_cut_speed", _("Max cut speed"), "int"),
    ("acceleration", _("Acceleration"), "int"),
    ("origin", _("Coordinate origin"), "string"),
    ("head_type", _("Head type"), "head_kind"),
    ("max_power", _("Head max power (S-value)"), "head_laser"),
    ("max_rpm", _("Head max RPM"), "head_spindle"),
    ("min_rpm", _("Head min RPM"), "head_spindle"),
    ("spot_size_mm", _("Spot size (X, Y)"), "head_laser_tuple2"),
    ("pwm_frequency", _("PWM frequency (Hz)"), "head_laser"),
    ("focal_distance", _("Focal distance"), "head_laser"),
    ("home_on_start", _("Home on start"), "bool"),
]


def _format_value(value: Any, kind: str) -> str:
    if value is None:
        return ""
    if kind == "tuple2_mm":
        a, b = value
        return f"{a} × {b}"
    if kind == "head_laser_tuple2":
        a, b = value
        return f"{a} × {b}"
    if isinstance(value, bool):
        return _("Yes") if value else _("No")
    return str(value)


class AILookupPage(WizardPage):
    step_number = 6
    title = _("AI Spec Lookup")
    subtitle = _(
        "If your machine is a known commercial model, the AI can "
        "pre-fill specification values from the manufacturer's "
        "documentation."
    )

    # Sent once the user advances with accepted suggestions. Payload
    # is the dict of accepted suggestions.
    def __init__(self, wizard, **kwargs):
        self.suggestions_applied = Signal()
        self._pulse_source_id = None
        self._progress_bar: Optional[Gtk.ProgressBar] = None
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self._install_progress_bar()

        self.group = _makePreferencesGroup(
            title=_("Vendor &amp; Model"),
            description=_(
                "Enter the machine's vendor (manufacturer) and model "
                "name. The more specific, the better — e.g. "
                '"Sculpfun" / "S30 Pro".'
            ),
        )
        self.content.append(self.group)

        self.vendor_row = Adw.EntryRow(title=_("Vendor (e.g. Sculpfun)"))
        self.group.add(self.vendor_row)

        self.model_row = Adw.EntryRow(title=_("Model (e.g. S30 Pro)"))
        self.model_row.connect("notify::text", self._on_inputs_changed)
        self.vendor_row.connect("notify::text", self._on_inputs_changed)
        self.group.add(self.model_row)

        self.lookup_button = Gtk.Button(label=_("Look Up Specs"))
        self.lookup_button.add_css_class("suggested-action")
        self.lookup_button.connect("clicked", lambda *_: self._run_lookup())
        self.lookup_button.set_sensitive(False)

        self.banner = Adw.Bin()
        self.content.append(self.banner)

        self.suggestions_group = _makePreferencesGroup(
            title=_("Suggestions"),
            description=_(
                "Suggested values are switched on; turn off any you "
                "don't want applied."
            ),
        )
        self.suggestions_group.set_visible(False)
        self.content.append(self.suggestions_group)

        self._rows: List[Adw.SwitchRow] = []
        self._accepted: Dict[str, Any] = {}

        # If AI isn't configured, surface a friendly message instead.
        if not is_ai_configured():
            self._show_not_configured_banner()
        else:
            self.banner.set_child(None)

        # Always ready: the user can skip even when no AI is set.
        self.set_ready(True)

    def _install_progress_bar(self) -> None:
        """Pin a thin pulse bar under the scrollable content, mirroring
        the AI workpiece generator dialog's busy affordance."""
        scrolled = self.get_child()
        if scrolled is None:
            return
        scrolled.set_vexpand(True)
        # Unparent the scrolled window first: re-appending a widget
        # that is still the page's child would fail with a GTK
        # "child already has a parent" critical and orphan the content.
        self.set_child(None)
        outer = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        outer.append(scrolled)
        self._progress_bar = Gtk.ProgressBar(hexpand=True)
        self._progress_bar.add_css_class("thin-progress-bar")
        self._progress_bar.set_visible(False)
        outer.append(self._progress_bar)
        self.set_child(outer)

        provider = Gtk.CssProvider()
        provider.load_from_string(
            """
            progressbar.thin-progress-bar {
                min-height: 5px;
            }
            """
        )
        Gtk.StyleContext.add_provider_for_display(
            self.get_display(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _start_pulse(self) -> None:
        if self._progress_bar is None:
            return
        self._progress_bar.set_visible(True)
        self._progress_bar.pulse()
        self._pulse_source_id = GLib.timeout_add(100, self._on_pulse_timeout)

    def _stop_pulse(self) -> None:
        if self._pulse_source_id:
            GLib.source_remove(self._pulse_source_id)
            self._pulse_source_id = None
        if self._progress_bar is None:
            return
        self._progress_bar.set_visible(False)

    def _on_pulse_timeout(self) -> bool:
        if self._progress_bar is not None:
            self._progress_bar.pulse()
        return True

    def _show_not_configured_banner(self) -> None:
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
        )
        box.add_css_class("card")
        box.set_margin_top(6)
        label = Gtk.Label(
            label=_(
                "No AI provider is configured in Settings. "
                "Configure one to enable automatic spec lookup, "
                "or skip this step and enter the values by hand."
            ),
            wrap=True,
            xalign=0.0,
            hexpand=True,
        )
        label.set_margin_start(12)
        label.set_margin_end(12)
        label.set_margin_top(12)
        label.set_margin_bottom(12)
        box.append(label)
        self.banner.set_child(box)
        self.lookup_button.set_sensitive(False)

    def _on_inputs_changed(self, _entry, _param) -> None:
        has_input = bool(self.vendor_row.get_text()) or bool(
            self.model_row.get_text()
        )
        if is_ai_configured() and has_input:
            self.lookup_button.set_sensitive(True)
        else:
            self.lookup_button.set_sensitive(False)

    def enter(self, profile: DeviceProfile) -> None:
        # If we already have vendor/model metadata from a profile
        # selection, prefill.
        meta = profile.meta
        if meta.vendor:
            self.vendor_row.set_text(meta.vendor)
        if meta.model:
            self.model_row.set_text(meta.model)
        self._on_inputs_changed(None, None)
        # The page never auto-advances; the user must click "Next" /
        # "Skip" via the wizard footer.

    def footer_buttons(self) -> List[Gtk.Button]:
        return [self.lookup_button]

    # ----- async lookup --------------------------------------------------

    def _run_lookup(self) -> None:
        vendor = self.vendor_row.get_text().strip()
        model = self.model_row.get_text().strip()
        if not (vendor or model):
            return
        self.lookup_button.set_sensitive(False)
        self.lookup_button.set_label(_("Looking up…"))
        self._start_pulse()

        context = get_context()

        async def _coro(exec_ctx):
            return {
                "specs": await lookup_machine_specs(vendor, model, context),
                "vendor": vendor,
                "model": model,
            }

        task_mgr.add_coroutine(
            _coro,
            key="unified_wizard_ai_lookup",
            when_done=self._on_lookup_done,
        )

    def _on_lookup_done(self, task) -> None:
        def _update():
            self._stop_pulse()
            self.lookup_button.set_sensitive(True)
            self.lookup_button.set_label(_("Look Up Specs"))
            try:
                result = task.result()
            except Exception as exc:
                self._show_lookup_error(str(exc))
                return
            if task.get_status() != "completed":
                self._show_lookup_error(_("Lookup failed"))
                return
            specs: Dict[str, Any] = result.get("specs", {})
            if not specs:
                self._show_lookup_error(
                    _(
                        "The AI couldn't return specifications for "
                        "this machine. You can enter the values "
                        "manually in the next steps."
                    )
                )
                return
            self._render_suggestions(specs)
            self.suggestions_group.set_visible(True)

        task_mgr.schedule_on_main_thread(_update)

    def _show_lookup_error(self, message: str) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.add_css_class("card")
        label = Gtk.Label(label=message, wrap=True, xalign=0.0, hexpand=True)
        label.set_margin_start(12)
        label.set_margin_end(12)
        label.set_margin_top(12)
        label.set_margin_bottom(12)
        box.append(label)
        self.banner.set_child(box)

    def _render_suggestions(self, specs: Dict[str, Any]) -> None:
        for row in self._rows:
            self.suggestions_group.remove(row)
        self._rows.clear()
        self._accepted.clear()

        for key, label, kind in _FIELD_SPEC:
            if key not in specs:
                continue
            value = specs[key]
            if value is None:
                continue
            row = Adw.SwitchRow(
                title=label,
                subtitle=_("AI suggests: {value}").format(
                    value=_format_value(value, kind)
                ),
            )
            # Suggested values start accepted; the user toggles any off.
            row.set_active(True)
            self._accepted[key] = value
            row.connect(
                "notify::active", self._make_toggle_handler(key, value)
            )
            self.suggestions_group.add(row)
            self._rows.append(row)

    def _make_toggle_handler(self, key: str, value: Any):
        """Return a notify::active handler bound to this suggestion.

        ``notify::active`` delivers ``(switch, pspec)``, so plain
        lambda defaults (``k=key``) would be clobbered by the pspec
        argument; the nested function pins the captured values
        instead.
        """

        def _on_active_changed(switch, _pspec) -> None:
            self._on_suggestion_toggled(key, value, switch.get_active())

        return _on_active_changed

    def _on_suggestion_toggled(
        self, key: str, value: Any, active: bool
    ) -> None:
        if active:
            self._accepted[key] = value
        else:
            self._accepted.pop(key, None)

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        meta = profile.meta
        vendor = self.vendor_row.get_text().strip()
        model = self.model_row.get_text().strip()
        if vendor and not meta.vendor:
            meta.vendor = vendor
        if model and not meta.model:
            meta.model = model

        if not self._accepted:
            return True

        mc = profile.machine_config
        for key, value in self._accepted.items():
            if key == "axis_extents":
                try:
                    a, b = value
                    mc.axis_extents = (float(a), float(b))
                except (TypeError, ValueError):
                    continue
            elif key == "max_travel_speed":
                mc.max_travel_speed = int(value)
            elif key == "max_cut_speed":
                mc.max_cut_speed = int(value)
            elif key == "acceleration":
                mc.acceleration = int(value)
            elif key == "origin":
                try:
                    mc.origin = Origin(str(value).lower())
                except ValueError:
                    continue
            elif key == "home_on_start":
                mc.home_on_start = bool(value)
            elif key in (
                "head_type",
                "max_power",
                "max_rpm",
                "min_rpm",
                "spot_size_mm",
                "pwm_frequency",
                "focal_distance",
            ):
                self._merge_head_field(mc, key, value)

        self.suggestions_applied.send(self, accepted=self._accepted)
        return True

    # ----- head merge ----------------------------------------------------

    def _merge_head_field(self, mc, key: str, value: Any) -> None:
        head = mc.heads[0] if mc.heads else {}

        if key == "head_type":
            kind = str(value).lower()
            if "spindle" in kind:
                head["head_class"] = "SpindleHead"
            else:
                head["head_class"] = "LaserHead"
            head.setdefault("name", _("Main Head"))
        elif key == "max_power":
            head.setdefault("name", _("Main Head"))
            head["max_power"] = int(value)
        elif key == "max_rpm":
            head["max_rpm"] = int(value)
        elif key == "min_rpm":
            head["min_rpm"] = int(value)
        elif key == "spot_size_mm":
            try:
                a, b = value
                head["spot_size_mm"] = [float(a), float(b)]
            except (TypeError, ValueError):
                pass
        elif key == "pwm_frequency":
            head["pwm_frequency"] = int(value)
        elif key == "focal_distance":
            head["focal_distance"] = float(value)

        if mc.heads:
            mc.heads[0] = head
        else:
            mc.heads = [head]


__all__ = ["AILookupPage"]
