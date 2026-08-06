import logging
from gettext import gettext as _
from typing import Optional

from blinker import ANY, Signal
from gi.repository import Adw, GLib, Gtk

from ...context import get_context
from ...shared.units.definitions import Unit, get_unit
from .adwfix import ensure_spinrow_min_width

logger = logging.getLogger(__name__)


class _StrongSignal(Signal):
    """
    A blinker Signal that holds receivers strongly by default.

    Widget consumers often connect inline lambdas (e.g.
    ``row.value_changed.connect(lambda r: ...)``). Under blinker's
    default weak referencing such lambdas have no other strong
    reference and are collected immediately, silently never firing.
    Holding strongly matches the GTK ``connect()`` semantics callers
    are used to; the widget and its owning page are co-owned and reach
    the cyclic collector together.
    """

    def connect(self, receiver, sender: object = ANY, weak: bool = False):
        return super().connect(receiver, sender=sender, weak=weak)


# Uniform character width for every spin entry. ``Gtk.SpinButton`` would
# otherwise size itself from its *initial* value (and would need per-field
# range knowledge for dynamic bounds), producing inconsistent widths across
# rows. A single fixed width keeps every entry visually identical and is
# wide enough for the values used across the app (e.g. ``-10000.00``).
_SPINROW_WIDTH_CHARS = 10


class SpinRow(Adw.ActionRow):
    """
    A subclassable spin row: :class:`Adw.ActionRow` + :class:`Gtk.SpinButton`.

    ``Adw.SpinRow`` is declared final by libadwaita and cannot be
    subclassed, so this widget composes an ActionRow with an embedded
    SpinButton to provide a real widget class that other rows
    (e.g. :class:`UnitSpinRow`) can build on.

    Consumer signal wiring uses the blinker signal :attr:`value_changed`
    only; it fires on user edits but not on programmatic
    :meth:`set_value`. Pass ``debounce_ms > 0`` to coalesce rapid edits.
    """

    __gtype_name__ = "RayforgeSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        *,
        lower: float = 0.0,
        upper: float = 1e9,
        step_increment: float = 1.0,
        page_increment: Optional[float] = None,
        digits: int = 0,
        numeric: bool = False,
        value: Optional[float] = None,
        debounce_ms: int = 0,
    ):
        super().__init__(title=title, activatable=False)
        if subtitle:
            self.set_subtitle(subtitle)

        self._is_updating = False
        self._debounce_ms = debounce_ms
        self._debounce_timer_id: Optional[int] = None
        self._last_emitted_value: Optional[float] = None

        adj = Gtk.Adjustment(
            lower=lower,
            upper=upper,
            step_increment=step_increment,
            page_increment=(
                step_increment * 10
                if page_increment is None
                else page_increment
            ),
            value=(lower if value is None else value),
        )
        self._spin_button = Gtk.SpinButton(adjustment=adj, digits=digits)
        self._spin_button.set_valign(Gtk.Align.CENTER)
        # Use one uniform entry width so every row is visually consistent.
        self._spin_button.set_width_chars(_SPINROW_WIDTH_CHARS)
        if numeric:
            self._spin_button.set_numeric(True)
        self._spin_button.connect("value-changed", self._on_value_changed)
        # Mirror the historical Adw.SpinRow wiring: value-changed alone does
        # not fire on every keystroke, so also observe notify::text to keep
        # live consumers (e.g. array previews) responsive while typing.
        self._spin_button.connect("notify::text", self._on_text_changed)

        self.add_suffix(self._spin_button)

        self.value_changed = _StrongSignal()
        self._destroy_handler_id = self.connect("destroy", self._on_destroy)

        ensure_spinrow_min_width(self)

    def get_value(self) -> float:
        """Return the current value (display units), text-aware."""
        return self._get_display_value()

    def get_int_value(self) -> int:
        """Return the current value as an int, clamped to the range."""
        return int(round(self._get_display_value()))

    def set_value(self, value: float) -> None:
        """
        Set the value programmatically.

        This does not emit :attr:`value_changed`; only user edits do.
        """
        if self._is_updating:
            return
        self._is_updating = True
        try:
            self._spin_button.set_value(value)
        finally:
            self._is_updating = False

    def set_range(self, lower: float, upper: float) -> None:
        """Update the adjustment lower and upper bounds."""
        adj = self._spin_button.get_adjustment()
        adj.set_lower(lower)
        adj.set_upper(upper)

    def set_digits(self, digits: int) -> None:
        self._spin_button.set_digits(digits)

    def get_digits(self) -> int:
        return self._spin_button.get_digits()

    def set_numeric(self, numeric: bool) -> None:
        self._spin_button.set_numeric(numeric)

    def get_adjustment(self) -> Gtk.Adjustment:
        return self._spin_button.get_adjustment()

    def get_spin_button(self) -> Gtk.SpinButton:
        """Escape hatch for callers that need the raw SpinButton."""
        return self._spin_button

    def set_editable(self, editable: bool) -> None:
        self._spin_button.set_editable(editable)

    def get_editable(self) -> bool:
        return self._spin_button.get_editable()

    def set_width_chars(self, n: int) -> None:
        self._spin_button.set_width_chars(n)

    def _get_display_value(self) -> float:
        # A keyboard edit may not be reflected in get_value() immediately
        # (the historical Adw.SpinRow bug), so prefer the editable text and
        # clamp to the adjustment range.
        adj = self._spin_button.get_adjustment()
        try:
            v = float(self._spin_button.get_text())
        except ValueError:
            v = float(self._spin_button.get_value())
        return max(adj.get_lower(), min(v, adj.get_upper()))

    def _on_value_changed(self, _spin_button: Gtk.SpinButton) -> None:
        if self._is_updating:
            return
        self._emit_changed()

    def _on_text_changed(self, _spin_button, _pspec) -> None:
        if self._is_updating:
            return
        self._emit_changed()

    def _emit_changed(self) -> None:
        # value-changed and notify::text both fire for a single edit; dedupe
        # by value so consumers see exactly one notification per real change.
        current = self._get_display_value()
        if (
            self._last_emitted_value is not None
            and abs(current - self._last_emitted_value) < 1e-12
        ):
            return
        self._last_emitted_value = current
        if self._debounce_ms > 0:
            if self._debounce_timer_id is not None:
                GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = GLib.timeout_add(
                self._debounce_ms, self._flush_changed
            )
        else:
            self.value_changed.send(self)

    def _flush_changed(self) -> bool:
        self._debounce_timer_id = None
        self.value_changed.send(self)
        return GLib.SOURCE_REMOVE

    def _on_destroy(self, _widget) -> None:
        if self._debounce_timer_id is not None:
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = None
        self._destroy_handler_id = None


class AngleSpinRow(SpinRow):
    """
    A spin row for angle values in degrees.

    Builds on :class:`SpinRow` with degree-appropriate defaults: a full
    rotation range of -360..360 degrees, whole-degree stepping and one
    decimal place. Pass ``lower``/``upper``/``digits`` to override
    (e.g. a 0..180 half-turn or an integer-degree field).
    """

    __gtype_name__ = "RayforgeAngleSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        *,
        lower: float = -360.0,
        upper: float = 360.0,
        digits: int = 1,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            lower=lower,
            upper=upper,
            digits=digits,
            **kwargs,
        )


class UnitSpinRow(SpinRow):
    """
    A unit-aware spin row.

    Builds on :class:`SpinRow`, showing the current unit (e.g. ``"mm"``)
    as a tooltip on the entry box, live conversion on display-unit
    changes, and base-unit get/set. The unit is shown via the tooltip
    rather than repeated in every subtitle or as a suffix.

    Values are exchanged with the caller in application base units through
    :meth:`get_value_in_base_units` / :meth:`set_value_in_base_units`.
    """

    __gtype_name__ = "RayforgeUnitSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        *,
        quantity: str = "length",
        lower: float = 0.0,
        upper: float = 1e9,
        step_increment: float = 1.0,
        page_increment: Optional[float] = None,
        digits: int = 2,
        numeric: bool = False,
        value_in_base: Optional[float] = None,
        min_value_in_base: Optional[float] = None,
        max_value_in_base: Optional[float] = None,
        debounce_ms: int = 0,
    ):
        super().__init__(
            title,
            subtitle,
            lower=lower,
            upper=upper,
            step_increment=step_increment,
            page_increment=page_increment,
            digits=digits,
            numeric=numeric,
            debounce_ms=debounce_ms,
        )

        self.quantity = quantity
        self._unit: Optional[Unit] = None
        self._min_digits = digits
        self._min_value_in_base = min_value_in_base
        self._max_value_in_base = max_value_in_base

        self._config_handler_id = get_context().config.changed.connect(
            self._on_config_changed
        )

        # Guard the initial unit/value setup so it does not fire
        # ``value_changed``.
        self._is_updating = True
        try:
            self.update_unit_and_bounds()
            if value_in_base is not None and self._unit:
                self._spin_button.set_value(
                    self._unit.from_base(value_in_base)
                )
        finally:
            self._is_updating = False

    def update_unit_and_bounds(self) -> None:
        """
        Re-read the active unit from config and refresh the unit tooltip,
        adjustment bounds, and digits.

        Does not touch the current value and does not manage the
        ``_is_updating`` guard; callers wrap as needed.
        """
        config = get_context().config
        unit_name = config.unit_preferences.get(self.quantity)
        self._unit = get_unit(unit_name) if unit_name else None
        if not self._unit:
            logger.warning(
                "UnitSpinRow: no unit found for quantity %r", self.quantity
            )
            return

        self._spin_button.set_tooltip_text(
            _("Value in {unit}").format(unit=self._unit.label)
        )

        adj = self._spin_button.get_adjustment()
        if self._max_value_in_base is not None:
            adj.set_upper(self._unit.from_base(self._max_value_in_base))
        if self._min_value_in_base is not None:
            adj.set_lower(self._unit.from_base(self._min_value_in_base))
        self._spin_button.set_digits(
            max(self._unit.precision, self._min_digits)
        )

    def get_value_in_base_units(self) -> float:
        """Return the current value converted to application base units."""
        if not self._unit:
            return self._get_display_value()
        return float(self._unit.to_base(self._get_display_value()))

    def set_value_in_base_units(self, base_value: float) -> None:
        """Set the value from an application base-unit value."""
        if self._is_updating:
            return
        self._is_updating = True
        try:
            self.update_unit_and_bounds()
            if not self._unit:
                logger.warning("UnitSpinRow: skipping set, no unit")
                return
            self._spin_button.set_value(self._unit.from_base(base_value))
        finally:
            self._is_updating = False

    def set_bounds_in_base(
        self,
        min_value_in_base: Optional[float],
        max_value_in_base: Optional[float],
    ) -> None:
        """Update the adjustment bounds (in base units) and re-render."""
        self._min_value_in_base = min_value_in_base
        self._max_value_in_base = max_value_in_base
        self.update_unit_and_bounds()

    def set_min_digits(self, min_digits: int) -> None:
        """Override the minimum number of decimal digits shown."""
        self._min_digits = min_digits
        self.update_unit_and_bounds()

    def _on_config_changed(self, _sender, **_kwargs) -> None:
        # Preserve the semantic value across a display-unit switch.
        if not self._unit:
            self.update_unit_and_bounds()
            return
        base_value = self._unit.to_base(self._get_display_value())
        self._is_updating = True
        try:
            self.update_unit_and_bounds()
            if self._unit:
                display_value = self._unit.from_base(base_value)
                if abs(display_value - self._get_display_value()) >= 1e-12:
                    self._spin_button.set_value(display_value)
        finally:
            self._is_updating = False

    def _on_destroy(self, _widget) -> None:
        super()._on_destroy(_widget)
        if self._config_handler_id:
            get_context().config.changed.disconnect(self._config_handler_id)
        self._config_handler_id = None


class LengthSpinRow(UnitSpinRow):
    """Unit-aware spin row for the ``length`` quantity (base unit mm)."""

    __gtype_name__ = "RayforgeLengthSpinRow"

    def __init__(self, title: str, subtitle: Optional[str] = None, **kwargs):
        super().__init__(title, subtitle, quantity="length", **kwargs)


class SpeedSpinRow(UnitSpinRow):
    """Unit-aware spin row for the ``speed`` quantity (base mm/min)."""

    __gtype_name__ = "RayforgeSpeedSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        *,
        step_increment: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            quantity="speed",
            step_increment=step_increment,
            **kwargs,
        )


class AccelerationSpinRow(UnitSpinRow):
    """Unit-aware spin row for ``acceleration`` (base mm/s^2)."""

    __gtype_name__ = "RayforgeAccelerationSpinRow"

    def __init__(
        self,
        title: str,
        subtitle: Optional[str] = None,
        *,
        step_increment: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            title,
            subtitle,
            quantity="acceleration",
            step_increment=step_increment,
            **kwargs,
        )
