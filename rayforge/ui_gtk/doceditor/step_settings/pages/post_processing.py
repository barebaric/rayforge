"""Post-processing transformers settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk

from rayforge.context import get_context
from rayforge.core.step import Step
from rayforge.pipeline.transformer import OpsTransformer
from rayforge.pipeline.transformer.placeholder import PlaceholderTransformer
from rayforge.ui_gtk.doceditor.step_settings.groups import (
    PlaceholderSettingsGroup,
    TransformerSettingsGroup,
)
from rayforge.ui_gtk.icons import get_icon
from rayforge.ui_gtk.shared.preferences_page import TrackedPreferencesPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class PostProcessingPage(TrackedPreferencesPage):
    """A page for the post-processing transformers of a Step."""

    use_expanders = True

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()
        self.editor = editor
        self.step = step
        producer_type = step.ASSEMBLER_NAME or "unknown"
        producer_key = producer_type.lower()
        self.key = f"{producer_key}/post-processing"
        self.path_prefix = "/step-settings/"

        self._main_group = Adw.PreferencesGroup()
        super().add(self._main_group)
        self._has_expanders = False

        all_transformer_dicts = (
            step.per_workpiece_transformers_dicts or []
        ) + (step.per_step_transformers_dicts or [])

        # Deduplicate by object identity (same dict can be in both lists)
        seen_ids = set()
        unique_transformer_dicts = []
        for t_dict in all_transformer_dicts:
            dict_id = id(t_dict)
            if dict_id not in seen_ids:
                seen_ids.add(dict_id)
                unique_transformer_dicts.append(t_dict)

        context = get_context()
        if context:
            for t_dict in unique_transformer_dicts:
                transformer = OpsTransformer.from_dict(t_dict)
                context.plugin_mgr.hook.transformer_settings_loaded(
                    dialog=self, step=step, transformer=transformer
                )
                # Add placeholder widget if transformer is not available
                if isinstance(transformer, PlaceholderTransformer):
                    self.add(
                        PlaceholderSettingsGroup(
                            editor,
                            transformer.label,
                            transformer,
                            self,
                            step,
                        )
                    )

        if not self._has_expanders:
            placeholder_label = Gtk.Label(
                label=_("No post-processing options available for this step."),
                halign=Gtk.Align.CENTER,
                margin_top=24,
                margin_bottom=24,
                wrap=True,
            )
            placeholder_label.add_css_class("dim-label")
            self._main_group.add(placeholder_label)

    def add(self, group):
        rows = getattr(group, "_rows", None)
        if rows is None:
            super().add(group)
            return

        title = group.get_title()
        subtitle = group.get_description()

        expander = Adw.ExpanderRow(title=title or "")
        if subtitle:
            expander.set_subtitle(subtitle)
        expander.set_expanded(False)

        warning_icon = get_icon("warning-symbolic")
        warning_icon.set_valign(Gtk.Align.CENTER)
        expander.add_prefix(warning_icon)

        def _update_warning_icon(grp=group, ico=warning_icon):
            unsupported = (
                isinstance(grp, TransformerSettingsGroup)
                and grp.is_unsupported()
            )
            ico.set_visible(unsupported)

        enable_switch_row = None
        for row in rows:
            if isinstance(row, Adw.SwitchRow) and enable_switch_row is None:
                enable_switch_row = row
                switch = Gtk.Switch()
                switch.set_active(row.get_active())
                switch.set_valign(Gtk.Align.CENTER)
                expander.add_suffix(switch)

                def _on_header_toggled(sw, pspec, orig=row):
                    if orig.get_active() != sw.get_active():
                        orig.set_active(sw.get_active())

                switch.connect("notify::active", _on_header_toggled)

                def _on_orig_toggled(r, pspec, sw=switch):
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
