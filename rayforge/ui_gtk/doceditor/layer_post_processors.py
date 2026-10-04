"""Layer-level post processor settings for the layer settings dialog.

A layer's post processors run on the layer's merged toolpath, in
machine space, after world→machine and WCS transforms. The machine
can define defaults (e.g. bed mesh correction applied to every job);
this group shows the effective state and lets the layer override,
customize, or disable each default per transformer name.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from ...pipeline.transformer import OpsTransformer
from ...pipeline.transformer.placeholder import PlaceholderTransformer
from .post_processor.groups import PlaceholderSettingsGroup
from .post_processor.registry import transformer_widget_registry

if TYPE_CHECKING:
    from ...core.layer import Layer
    from ...doceditor.editor import DocEditor
    from ...machine.models.machine import Machine


class LayerPostProcessorGroup(Adw.PreferencesGroup):
    """Edits a layer's post processors against the machine defaults.

    The group is self-contained: state changes are persisted through
    the editor's undoable :meth:`set_layer_post_processors` command
    (or directly on the layer when no editor is available), and the
    UI rebuilds from the layer's dicts afterwards.
    """

    use_expanders = True

    def __init__(self, layer: "Layer", editor: "DocEditor | None"):
        super().__init__(
            title=_("Post Processing"),
            description=_(
                "Transformers applied to this layer's merged toolpath."
            ),
        )
        self.layer = layer
        self.editor = editor
        self._group_dicts: dict[Any, dict] = {}
        self._children: list[Gtk.Widget] = []
        self._build()

    # ── State helpers ─────────────────────────────────────────────

    def _machine(self) -> "Machine | None":
        from ...context import get_context

        return get_context().machine

    def _save(self, dicts: list[dict]) -> None:
        if self.editor is not None:
            self.editor.layer.set_layer_post_processors(self.layer, dicts)
        else:
            self.layer.set_post_processors_dicts(dicts)
        self._rebuild()

    # ── UI construction ───────────────────────────────────────────

    def _build(self):
        for child in self._children:
            self.remove(child)
        self._children.clear()
        self._group_dicts.clear()

        for t_dict in self.layer.post_processors_dicts:
            transformer = OpsTransformer.from_dict(t_dict)
            widget_cls = transformer_widget_registry.get(type(transformer))
            if widget_cls is not None:
                group = widget_cls(
                    transformer.label,
                    transformer,
                    self,
                )
                self._add_group(group, t_dict)
            elif isinstance(transformer, PlaceholderTransformer):
                group = PlaceholderSettingsGroup(
                    transformer.label,
                    transformer,
                    self,
                )
                self.add(group)

        machine = self._machine()
        defaults = machine.default_post_processors_dicts if machine else []
        default_names = {
            d.get("name") for d in defaults if d.get("enabled", True)
        }
        own_names = {
            d.get("name")
            for d in self.layer.post_processors_dicts
            if d.get("name")
        }
        if default_names - own_names:
            row = Adw.ActionRow(
                title=_("Machine Defaults Apply"),
                subtitle=_(
                    "This layer uses the machine's default post processors."
                ),
            )
            for name in sorted(
                n for n in default_names - own_names if n is not None
            ):
                override_btn = Gtk.Button(
                    label=_("Customize"), valign=Gtk.Align.CENTER
                )
                override_btn.connect(
                    "clicked",
                    lambda _b, n=name: self._override_default(n),
                )
                row.add_suffix(override_btn)
                disable_btn = Gtk.Button(
                    label=_("Disable"),
                    valign=Gtk.Align.CENTER,
                    css_classes=["destructive-action"],
                )
                disable_btn.connect(
                    "clicked",
                    lambda _b, n=name: self._disable_default(n),
                )
                row.add_suffix(disable_btn)
            self.add(row)

    def _add_group(self, group, t_dict: dict):
        """Wrap a transformer settings group in an expander row.

        Mirrors the step settings page: the group's rows move into an
        ``Adw.ExpanderRow`` with the group's enable switch as suffix.
        """
        title = group.get_title()
        subtitle = group.get_description()

        expander = Adw.ExpanderRow(title=title or "")
        if subtitle:
            expander.set_subtitle(subtitle)
        expander.set_expanded(True)

        for row in group._rows:
            expander.add_row(row)
        switch = group.enable_switch
        if switch is not None:
            expander.add_suffix(switch)

        group.param_changed.connect(self._on_param_changed)
        self._group_dicts[group] = t_dict
        self.add(expander)

    def add(self, child: Gtk.Widget) -> None:
        super().add(child)
        self._children.append(child)

    def _rebuild(self):
        self._build()

    # ── Actions ───────────────────────────────────────────────────

    def _override_default(self, name: str):
        """Copy the machine default into the layer for customization."""
        machine = self._machine()
        if machine is None:
            return
        source = next(
            (
                d
                for d in machine.default_post_processors_dicts
                if d.get("name") == name
            ),
            None,
        )
        if source is None:
            return
        dicts = [
            d
            for d in self.layer.post_processors_dicts
            if d.get("name") != name
        ]
        dicts.append(dict(source))
        self._save(dicts)

    def _disable_default(self, name: str):
        dicts = [
            d
            for d in self.layer.post_processors_dicts
            if d.get("name") != name
        ]
        dicts.append({"name": name, "enabled": False})
        self._save(dicts)

    def reset_to_machine_defaults(self, name: str):
        """Drop the layer's entry for *name*, falling back to defaults."""
        dicts = [
            d
            for d in self.layer.post_processors_dicts
            if d.get("name") != name
        ]
        self._save(dicts)

    # ── Widget callbacks ──────────────────────────────────────────

    def _on_param_changed(
        self,
        group,
        *,
        key: str,
        value: Any,
        name: str,
    ):
        t_dict = self._group_dicts.get(group)
        if t_dict is None or t_dict.get(key) == value:
            return
        dicts = [
            dict(d) if d is t_dict else d
            for d in self.layer.post_processors_dicts
        ]
        updated = next(d for d in dicts if d.get("name") == t_dict.get("name"))
        updated[key] = value
        self._save(dicts)
