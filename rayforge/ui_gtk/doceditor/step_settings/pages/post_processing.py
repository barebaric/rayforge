"""Step-mode post-processing transformers settings page."""

from typing import TYPE_CHECKING, Any

from gi.repository import Adw, GObject, Gtk

from .....context import get_context
from .....core.step import Step
from ....icons import get_icon
from ...post_processor.groups import TransformerSettingsGroup
from ...post_processor.page_base import PostProcessingPageBase

if TYPE_CHECKING:
    from .....doceditor.editor import DocEditor


class PostProcessingPage(PostProcessingPageBase):
    """A page for the post-processing transformers of a Step.

    The transformer widgets are pure UI: they announce parameter
    changes via ``param_changed`` and the page persists them through
    the editor's undoable command path (``editor.step.set_step_param``).
    """

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()
        self.editor = editor
        self.step = step
        producer_type = step.ASSEMBLER_NAME or "unknown"
        producer_key = producer_type.lower()
        self.key = f"{producer_key}/post-processing"
        self.path_prefix = "/step-settings/"

        all_transformer_dicts = (
            step.per_workpiece_transformers_dicts or []
        ) + (step.per_step_transformers_dicts or [])

        self.populate(all_transformer_dicts)

    def _step_for_hook(self) -> "Step | None":
        return self.step

    def _add_group(
        self,
        group: TransformerSettingsGroup,
        t_dict: dict,
    ) -> None:
        group.param_changed.connect(self._on_param_changed)

        title = group.get_title()
        subtitle = group.get_description()
        rows = group._rows

        expander = Adw.ExpanderRow(title=title or "")
        if subtitle:
            expander.set_subtitle(subtitle)
        expander.set_expanded(False)

        warning_icon = get_icon("warning-symbolic")
        warning_icon.set_valign(Gtk.Align.CENTER)
        expander.add_prefix(warning_icon)

        def _update_warning_icon(
            grp: TransformerSettingsGroup = group,
            ico: Gtk.Image = warning_icon,
        ) -> None:
            ico.set_visible(grp.is_unsupported())

        enable_switch_row: Adw.SwitchRow | None = None
        for row in rows:
            if isinstance(row, Adw.SwitchRow) and enable_switch_row is None:
                enable_switch_row = row
                switch = Gtk.Switch()
                switch.set_active(row.get_active())
                switch.set_valign(Gtk.Align.CENTER)
                expander.add_suffix(switch)

                def _on_header_toggled(
                    sw: Gtk.Switch,
                    _pspec: GObject.ParamSpec,
                    orig: Adw.SwitchRow = row,
                ) -> None:
                    if orig.get_active() != sw.get_active():
                        orig.set_active(sw.get_active())

                switch.connect("notify::active", _on_header_toggled)

                def _on_orig_toggled(
                    r: Adw.SwitchRow,
                    _pspec: GObject.ParamSpec,
                    sw: Gtk.Switch = switch,
                ) -> None:
                    if sw.get_active() != r.get_active():
                        sw.set_active(r.get_active())

                row.connect("notify::active", _on_orig_toggled)
                row.connect(
                    "notify::active",
                    lambda *_: _update_warning_icon(),
                )
            else:
                expander.add_row(row)

        machine = get_context().machine
        if machine:
            machine.changed.connect(lambda *_: _update_warning_icon())

        _update_warning_icon()
        self._main_group.add(expander)
        self._has_expanders = True

    def _on_param_changed(
        self,
        group: TransformerSettingsGroup,
        *,
        key: str,
        value: Any,
        name: str,
    ) -> None:
        """Persist a widget's announced change via the editor."""
        t_dict = self._group_dicts[group]
        if t_dict in self.step.per_step_transformers_dicts:
            callback = self.step.per_step_transformer_changed.send
        else:
            callback = self._send_step_updated
        self.editor.step.set_step_param(
            target_dict=t_dict,
            key=key,
            new_value=value,
            name=name,
            on_change_callback=callback,
        )

    def _send_step_updated(self) -> None:
        self.step.updated.send(self.step)
