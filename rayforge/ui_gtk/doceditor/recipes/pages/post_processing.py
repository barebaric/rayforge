"""Recipe-mode post-processing transformers settings page."""

from __future__ import annotations

from gettext import gettext as _
from typing import Any

from gi.repository import Adw, Gtk

from .....pipeline.transformer import OpsTransformer
from .....pipeline.transformer.placeholder import PlaceholderTransformer
from ....shared.preferences_page import TrackedPreferencesPage
from ...post_processor.groups import (
    PlaceholderSettingsGroup,
    TransformerSettingsGroup,
)
from ...post_processor.registry import transformer_widget_registry


class RecipePostProcessingPage(TrackedPreferencesPage):
    """A page for editing transformer settings stored on a recipe.

    Unlike the step-mode page this one has no editor or step: it owns
    the transformer dicts and mutates them directly when the widgets
    announce changes. Each group is wrapped in an
    :class:`Adw.ExpanderRow` whose header carries the group's apply
    toggle as a prefix and its enable switch as a suffix, matching the
    step-mode page:

    - **Toggle off** (``recipe_apply=False``): the recipe will not
      touch this transformer when applied.
    - **Toggle on** (``recipe_apply=True``): the recipe stamps the
      transformer's params, including its native enable switch, which
      lives in the header like in the step settings dialog.
    """

    use_expanders = True

    def __init__(self, transformer_dicts: list[dict[str, Any]] | None = None):
        super().__init__()
        self.key = "post-processing"
        self.path_prefix = "/recipe/"

        self._main_group = Adw.PreferencesGroup(
            title=_("Post Processing"),
            description=_(
                "Transformer settings applied by this recipe. When multiple "
                "step types are selected, only transformers common to "
                "all of them are shown."
            ),
        )
        self.add(self._main_group)
        self._group_dicts: dict[TransformerSettingsGroup, dict] = {}
        self._has_expanders = False
        self.populate(transformer_dicts or [])

    # -- Public API -----------------------------------------------------

    def get_transformer_dicts(self) -> list[dict[str, Any]]:
        """Return the (possibly mutated) transformer dicts."""
        return list(self._group_dicts.values())

    # -- Construction ---------------------------------------------------

    def populate(self, transformer_dicts: list[dict[str, Any]]) -> None:
        """Build groups for the given transformer dicts."""
        # Deduplicate by object identity (same dict can be in both lists)
        seen_ids: set[int] = set()
        unique_transformer_dicts: list[dict[str, Any]] = []
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
                    apply_toggle=True,
                    initial_apply=bool(t_dict.get("recipe_apply", False)),
                )
            elif isinstance(transformer, PlaceholderTransformer):
                group = PlaceholderSettingsGroup(
                    transformer.label,
                    transformer,
                    self,
                    apply_toggle=True,
                    initial_apply=bool(t_dict.get("recipe_apply", False)),
                )
            else:
                continue
            self._group_dicts[group] = t_dict
            self._add_group(group, t_dict)

        if not self._has_expanders:
            self._show_empty_state()

    def _add_group(
        self,
        group: TransformerSettingsGroup,
        t_dict: dict,
    ) -> None:
        title = group.get_title()
        subtitle = group.get_description()

        expander = Adw.ExpanderRow(title=title or "")
        if subtitle:
            expander.set_subtitle(subtitle)
        expander.set_expanded(False)

        for row in group._rows:
            expander.add_row(row)

        switch = group.enable_switch
        if switch is not None:
            expander.add_suffix(switch)

        toggle = group.apply_toggle
        if toggle is not None:
            toggle.set_valign(Gtk.Align.CENTER)
            expander.add_prefix(toggle)

        group.param_changed.connect(self._on_param_changed)
        group.apply_changed.connect(self._on_apply_changed)

        self._main_group.add(expander)
        self._has_expanders = True

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

    # -- Change handlers ------------------------------------------------

    def _on_param_changed(
        self,
        group: TransformerSettingsGroup,
        *,
        key: str,
        value: Any,
        name: str,
    ) -> None:
        """Persist a widget's announced change via direct dict mutation."""
        t_dict = self._group_dicts.get(group)
        if t_dict is None:
            return
        t_dict[key] = value

    def _on_apply_changed(
        self, group: TransformerSettingsGroup, *, state: bool
    ) -> None:
        """Persist the apply toggle onto the backing dict."""
        t_dict = self._group_dicts.get(group)
        if t_dict is None:
            return
        t_dict["recipe_apply"] = bool(state)
