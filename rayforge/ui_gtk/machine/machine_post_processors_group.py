"""Machine-level default post processor configuration group.

Renders every registered layer-applicable transformer from the
machine's default post processor dicts — unconditionally. Whether a
transformer is usable right now (e.g. a probed bed mesh) is the
addon-provided widget's own concern. Interacting with a transformer
that has no entry yet claims one; entries are persisted through
:meth:`Machine.set_default_post_processors`.
"""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw, Gtk

from ...machine.models.machine import Machine
from ...pipeline.transformer import OpsTransformer
from ...pipeline.transformer.registry import transformer_registry
from ..doceditor.post_processor.registry import transformer_widget_registry


class MachinePostProcessorsGroup(Adw.PreferencesGroup):
    """Edits the machine's default post processors.

    The group is self-contained: changes persist through
    :meth:`Machine.set_default_post_processors`, which notifies the
    machine's change signal.
    """

    use_expanders = True

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Post Processing"),
            description=_(
                "Transformers applied to every job on this machine. "
                "Layers can override them in their settings."
            ),
            **kwargs,
        )
        self.machine = machine
        self._children: list[Gtk.Widget] = []
        #: widget group -> (transformer class, entry dict or None,
        #: source dict the widget was built from)
        self._group_dicts: dict[Any, tuple[type, dict | None, dict]] = {}
        self._build()

    def _build(self):
        for child in self._children:
            self.remove(child)
        self._children.clear()
        self._group_dicts.clear()

        for transformer_cls in transformer_registry.get_layer_applicable():
            name = transformer_cls.__name__
            widget_cls = transformer_widget_registry.get(transformer_cls)
            if widget_cls is None:
                continue
            entry = next(
                (
                    d
                    for d in self.machine.default_post_processors_dicts
                    if d.get("name") == name
                ),
                None,
            )
            source = entry or transformer_cls().to_dict()
            transformer = OpsTransformer.from_dict(dict(source))
            group = widget_cls(transformer.label, transformer, self)

            expander = Adw.ExpanderRow(title=group.get_title() or "")
            if group.get_description():
                expander.set_subtitle(group.get_description())
            expander.set_expanded(True)
            for row in group._rows:
                expander.add_row(row)
            switch = group.enable_switch
            if switch is not None:
                expander.add_suffix(switch)
            group.param_changed.connect(self._on_param_changed)

            self._group_dicts[group] = (transformer_cls, entry, dict(source))
            self.add(expander)

    def add(self, child: Gtk.Widget) -> None:
        super().add(child)
        self._children.append(child)

    def _on_param_changed(
        self,
        group,
        *,
        key: str,
        value: Any,
        name: str,
    ):
        transformer_cls, entry, source = self._group_dicts[group]
        dicts = [dict(d) for d in self.machine.default_post_processors_dicts]
        if entry is None:
            # First interaction on a transformer without an entry:
            # claim one seeded from the displayed configuration.
            dicts.append(dict(source))
        target = next(
            d for d in dicts if d.get("name") == transformer_cls.__name__
        )
        target[key] = value
        self.machine.set_default_post_processors(dicts)

        new_entry = next(
            d
            for d in self.machine.default_post_processors_dicts
            if d.get("name") == transformer_cls.__name__
        )
        self._group_dicts[group] = (transformer_cls, new_entry, source)
