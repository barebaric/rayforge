"""Base row wrapper for step settings.

A ``StepRow`` wraps an ``Adw.PreferencesRow`` widget (exposed as
``.widget``) because several Adw row types are final GTypes and
cannot be subclassed. Every row edits one step attribute and
re-syncs its value and dependent state whenever the step changes.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Optional

from gi.repository import GLib

from rayforge.core.undo import ChangePropertyCommand

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class StepRow:
    """Base wrapper for a row that edits one step attribute.

    Subclasses set ``attr`` to the step attribute they edit, build
    the underlying widget in ``build_widget``, and may override
    ``set_widget_value`` and ``_sync_dependencies``.
    """

    attr: str = ""

    def __init__(self, editor: "DocEditor", step: Any):
        self.editor = editor
        self.step = step
        self.history_manager = editor.history_manager
        self._syncing = False
        self.widget: Any = self.build_widget()
        step.updated.connect(self._on_step_updated)

    def build_widget(self) -> Any:
        raise NotImplementedError

    def _on_step_updated(self, *args):
        # Do not clobber an in-progress edit: a row with uncommitted
        # user input would otherwise be overwritten by the external
        # step change.
        if not self._has_pending_edit():
            self._sync_from_step()
        self._sync_dependencies()

    def _has_pending_edit(self) -> bool:
        """Whether the row has uncommitted user input."""
        return False

    def _sync_from_step(self):
        if not self.attr:
            return
        self._syncing = True
        try:
            self.set_widget_value(getattr(self.step, self.attr, None))
        finally:
            self._syncing = False

    def resync(self):
        """Force the widget to reflect the current step value.

        Unlike the ``updated``-driven sync, this ignores any pending
        (debounced) user edit so a recipe application or model reset is
        shown immediately.
        """
        cancel = getattr(self, "cancel_pending", None)
        if callable(cancel):
            cancel()
        self._sync_from_step()
        self._sync_dependencies()

    def set_widget_value(self, value: Any):
        pass

    def _sync_dependencies(self):
        pass

    def cleanup(self):
        pass

    def set_visible(self, visible: bool):
        self.widget.set_visible(visible)

    def set_sensitive(self, sensitive: bool):
        self.widget.set_sensitive(sensitive)

    def get_machine(self):
        return getattr(self.editor.context, "machine", None)

    def get_selected_head(self):
        machine = self.get_machine()
        if machine is None:
            return None
        return self.step.get_selected_head(machine)

    def commit(self, value: Any, name: Optional[str] = None):
        if not self.attr:
            return
        if getattr(self.step, self.attr, None) == value:
            return
        setter_name = f"set_{self.attr}"
        setter = getattr(self.step, setter_name, None)
        command = ChangePropertyCommand(
            target=self.step,
            property_name=self.attr,
            new_value=value,
            setter_method_name=setter_name if setter else None,
            name=name
            or _("Change {key}").format(key=self.attr.replace("_", " ")),
            on_change_callback=None
            if setter
            else lambda: self.step.updated.send(self.step),
        )
        self.history_manager.execute(command)


class DebouncedMixin:
    """Debounced commit helper for rows that fire frequent changes."""

    def __init__(self):
        self._debounce_timer = 0

    def _debounced(self, callback, *args):
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
        self._debounce_timer = GLib.timeout_add(
            300, self._fire_debounce, callback, args
        )

    def _fire_debounce(self, callback, args):
        self._debounce_timer = 0
        callback(*args)
        return GLib.SOURCE_REMOVE

    def _has_pending_edit(self) -> bool:
        return self._debounce_timer > 0

    def cancel_pending(self):
        """Cancel a scheduled debounce without committing it."""
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0

    def cleanup(self):
        self.cancel_pending()
