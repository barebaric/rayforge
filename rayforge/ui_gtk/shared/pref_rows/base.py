import logging

from blinker import ANY, Signal
from gi.repository import Adw, GLib, Gtk

from ..adwfix import ensure_row_min_width

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
        subtitle: str | None = None,
        *,
        lower: float = 0.0,
        upper: float = 1e9,
        step_increment: float = 1.0,
        page_increment: float | None = None,
        digits: int = 0,
        numeric: bool = False,
        value: float | None = None,
        debounce_ms: int = 0,
    ):
        super().__init__(title=title, activatable=False)
        if subtitle:
            self.set_subtitle(subtitle)

        self._is_updating = False
        self._debounce_ms = debounce_ms
        self._debounce_timer_id: int | None = None
        self._last_emitted_value: float | None = None

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

        ensure_row_min_width(self)

    def get_value(self) -> float:
        """Return the current value (display units), text-aware."""
        return self._get_display_value()

    def get_int_value(self) -> int:
        """Return the current value as an int, clamped to the range."""
        return round(self._get_display_value())

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
