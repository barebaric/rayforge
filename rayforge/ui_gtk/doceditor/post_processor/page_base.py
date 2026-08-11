"""Shared machinery for post-processing settings pages."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk

from ....pipeline.transformer import OpsTransformer
from ....pipeline.transformer.placeholder import PlaceholderTransformer
from ...shared.preferences_page import TrackedPreferencesPage
from .groups import (
    PlaceholderSettingsGroup,
    TransformerSettingsGroup,
)
from .registry import transformer_widget_registry

if TYPE_CHECKING:
    from ....core.step import Step


class PostProcessingPageBase(TrackedPreferencesPage):
    """Shared machinery for post-processing transformer settings pages.

    Owns the transformer-dict iteration, widget-class lookup via
    :data:`transformer_widget_registry`, placeholder fallback, and the
    empty-state message. Subclasses provide mode-specific behavior
    through :meth:`_step_for_hook` and :meth:`_add_group`.
    """

    use_expanders = True

    def __init__(self):
        super().__init__()
        self._main_group = Adw.PreferencesGroup()
        self.add(self._main_group)
        self._has_expanders = False
        self._group_dicts: dict[TransformerSettingsGroup, dict] = {}

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

        step = self._step_for_hook()
        for t_dict in unique_transformer_dicts:
            transformer = OpsTransformer.from_dict(t_dict)
            widget_cls = transformer_widget_registry.get(type(transformer))
            if widget_cls:
                group = widget_cls(
                    transformer.label,
                    transformer,
                    self,
                    step=step,
                )
            elif isinstance(transformer, PlaceholderTransformer):
                group = PlaceholderSettingsGroup(
                    transformer.label,
                    transformer,
                    self,
                    step=step,
                )
            else:
                continue
            self._group_dicts[group] = t_dict
            self._add_group(group, t_dict)

        if not self._has_expanders:
            self._show_empty_state()

    def _step_for_hook(self) -> "Step | None":
        """The step providing read-only context for the widgets, if any."""
        return None

    def _add_group(
        self,
        group: TransformerSettingsGroup,
        t_dict: dict,
    ) -> None:
        """Wrap a group in the page's layout and connect its signal."""
        raise NotImplementedError

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
