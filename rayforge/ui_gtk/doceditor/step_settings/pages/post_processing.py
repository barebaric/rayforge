"""Step-mode post-processing transformers settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from .....context import get_context
from .....core.step import Step
from .....pipeline.transformer import OpsTransformer
from .....pipeline.transformer.placeholder import PlaceholderTransformer
from ....icons import get_icon
from ....shared.preferences_page import TrackedPreferencesPage
from ...post_processor.groups import (
    PlaceholderSettingsGroup,
    TransformerSettingsGroup,
)
from ...post_processor.registry import transformer_widget_registry

if TYPE_CHECKING:
    from .....doceditor.editor import DocEditor


class PostProcessingPage(TrackedPreferencesPage):
    """A page for the post-processing transformers of a Step.

    The transformer widgets are pure UI: they announce parameter
    changes via ``param_changed`` and the page persists them through
    the editor's undoable command path (``editor.step.set_step_param``).
    """

    use_expanders = True

    def __init__(self, editor: "DocEditor", step: Step):
        super().__init__()
        self.editor = editor
        self.step = step
        producer_type = step.ASSEMBLER_NAME or "unknown"
        producer_key = producer_type.lower()
        self.key = f"{producer_key}/post-processing"
        self.path_prefix = "/step-settings/"

        self._main_group = Adw.PreferencesGroup(
            title=_("Post Processing"),
            description=_(
                "Transformers applied to this step's generated toolpath."
            ),
        )
        self.add(self._main_group)
        self._has_expanders = False
        self._group_dicts: dict[TransformerSettingsGroup, dict] = {}

        all_transformer_dicts = (
            step.per_workpiece_transformers_dicts or []
        ) + (step.per_step_transformers_dicts or [])

        self.populate(all_transformer_dicts)

    def populate(self, transformer_dicts: list[dict]) -> None:
        """Build groups for the given transformer dicts."""
        # Deduplicate by object identity (same dict can be in both lists)
        seen_ids: set[int] = set()
        unique_transformer_dicts: list[dict] = []
        for t_dict in transformer_dicts or []:
            dict_id = id(t_dict)
            if dict_id not in seen_ids:
                seen_ids.add(dict_id)
                unique_transformer_dicts.append(t_dict)

        for t_dict in unique_transformer_dicts:
            transformer = OpsTransformer.from_dict(t_dict)
            widget_cls = transformer_widget_registry.get(type(transformer))
            if widget_cls:
                group = widget_cls(
                    transformer.label,
                    transformer,
                    self,
                    step=self.step,
                )
            elif isinstance(transformer, PlaceholderTransformer):
                group = PlaceholderSettingsGroup(
                    transformer.label,
                    transformer,
                    self,
                    step=self.step,
                )
            else:
                continue
            self._group_dicts[group] = t_dict
            self._add_group(group, t_dict)

        if not self._has_expanders:
            self._show_empty_state()

    def _show_empty_state(self) -> None:
        """Render the empty-state message when no groups were added."""
        placeholder_label = Gtk.Label(
            label=_("No post-processing options available for this step."),
            halign=Gtk.Align.CENTER,
            margin_top=24,
            margin_bottom=24,
            wrap=True,
        )
        placeholder_label.add_css_class("dim-label")
        self._main_group.add(placeholder_label)

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

        enable_switch = group.enable_switch
        for row in rows:
            expander.add_row(row)
        if enable_switch is not None:
            switch = enable_switch
            expander.add_suffix(switch)
            switch.connect(
                "notify::active",
                lambda *_: _update_warning_icon(),
            )

        machine = get_context().machine
        if machine:
            machine.changed.connect(
                lambda *_: _update_warning_icon(),
                weak=False,
            )

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
