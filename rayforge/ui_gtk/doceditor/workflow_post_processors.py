"""Workflow-level post processor settings.

A workflow's post processors run on the workflow's merged toolpath,
in machine space, after world→machine and WCS transforms. Every
registered workflow-applicable transformer is rendered here
unconditionally — whether it is usable right now (e.g. whether a bed
mesh has been probed) is the addon-provided widget's own concern; it
banners and/or insensitizes itself.

Each rendered group reflects the effective configuration: the
workflow's own entry if it has one, otherwise the machine default,
otherwise the transformer's defaults. Interacting with a group that
is not backed by a workflow entry claims one (lazily, on first
change), so a workflow can customize or disable a machine default per
transformer name.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from ...pipeline.transformer import OpsTransformer
from ...pipeline.transformer.registry import transformer_registry
from .post_processor.registry import transformer_widget_registry

if TYPE_CHECKING:
    from ...core.workflow import Workflow
    from ...doceditor.editor import DocEditor
    from ...machine.models.machine import Machine


class WorkflowPostProcessorGroup(Adw.PreferencesGroup):
    """Edits a workflow's post processors against the machine defaults.

    The group is self-contained: state changes are persisted through
    the editor's undoable commands (or directly on the workflow when
    no editor is available), and the UI rebuilds from the workflow's
    dicts when the entry set itself changes.
    """

    use_expanders = True

    def __init__(self, workflow: "Workflow", editor: "DocEditor | None"):
        super().__init__(
            title=_("Post Processing"),
            description=_(
                "Transformers applied to this workflow's merged toolpath."
            ),
        )
        self.workflow = workflow
        self.editor = editor
        self._children: list[Gtk.Widget] = []
        #: widget group -> (transformer class, owned dict or None,
        #: source dict the widget was built from)
        self._group_dicts: dict[Any, tuple[type, dict | None, dict]] = {}
        self._build()

    # ── State helpers ─────────────────────────────────────────────

    def _machine(self) -> "Machine | None":
        from ...context import get_context

        return get_context().machine

    def _owned_entry(self, name: str) -> dict | None:
        return next(
            (
                d
                for d in self.workflow.post_processors_dicts
                if d.get("name") == name
            ),
            None,
        )

    def _persist(self, dicts: list[dict]):
        if self.editor is not None:
            self.editor.workflow.set_workflow_post_processors(
                self.workflow, dicts
            )
        else:
            self.workflow.set_post_processors_dicts(dicts)

    def _save_param(self, target_dict: dict, key: str, value: Any):
        """Persist one parameter of an owned entry, undoable."""
        if self.editor is not None:
            self.editor.workflow.set_workflow_post_processor_param(
                target_dict,
                key,
                value,
                name=_("Change workflow post processor"),
            )
        else:
            target_dict[key] = value

    # ── UI construction ───────────────────────────────────────────

    def _build(self):
        for child in self._children:
            self.remove(child)
        self._children.clear()
        self._group_dicts.clear()

        machine = self._machine()
        defaults_by_name: dict[str, dict] = {
            d["name"]: d
            for d in (machine.default_post_processors_dicts if machine else [])
            if isinstance(d.get("name"), str)
        }

        for transformer_cls in transformer_registry.get_workflow_applicable():
            name = transformer_cls.__name__
            widget_cls = transformer_widget_registry.get(transformer_cls)
            if widget_cls is None:
                continue
            owned = self._owned_entry(name)
            source = (
                owned
                or defaults_by_name.get(name)
                or transformer_cls().to_dict()
            )
            transformer = OpsTransformer.from_dict(dict(source))
            group = widget_cls(transformer.label, transformer, self)
            self._add_group(
                group,
                transformer_cls,
                owned=owned,
                source=source,
                follows_default=(owned is None and name in defaults_by_name),
            )

    def _add_group(
        self,
        group,
        transformer_cls: type,
        *,
        owned: dict | None,
        source: dict,
        follows_default: bool,
    ):
        """Wrap a transformer settings group in an expander row.

        The group's rows move into an ``Adw.ExpanderRow`` with the
        enable switch as suffix. When the workflow does not own an
        entry but a machine default exists, the subtitle states that
        the default is being followed; interacting with the group
        claims an entry (lazily). An owned entry with a machine
        default gets a reset affordance.
        """
        name = transformer_cls.__name__
        expander = Adw.ExpanderRow(title=group.get_title() or "")
        if follows_default:
            expander.set_subtitle(_("Follows the machine default"))
        expander.set_expanded(True)

        for row in group._rows:
            expander.add_row(row)
        switch = group.enable_switch
        if switch is not None:
            expander.add_suffix(switch)

        machine = self._machine()
        if (
            owned is not None
            and machine is not None
            and any(
                d.get("name") == name
                for d in machine.default_post_processors_dicts
            )
        ):
            reset_btn = Gtk.Button(
                label=_("Reset to Machine Default"),
                valign=Gtk.Align.CENTER,
            )
            reset_btn.add_css_class("flat")
            reset_btn.connect(
                "clicked",
                lambda _b, c=transformer_cls: self._reset_to_default(c),
            )
            expander.add_suffix(reset_btn)

        group.param_changed.connect(self._on_param_changed)
        self._group_dicts[group] = (transformer_cls, owned, dict(source))
        self.add(expander)

    def add(self, child: Gtk.Widget) -> None:
        super().add(child)
        self._children.append(child)

    def _rebuild(self):
        self._build()

    # ── Actions ───────────────────────────────────────────────────

    def _claim_entry(self, transformer_cls: type, source: dict) -> dict:
        """Add an owned entry for *transformer_cls*, undoable.

        Returns the dict object now living in the workflow.
        """
        dicts = [
            d
            for d in self.workflow.post_processors_dicts
            if d.get("name") != transformer_cls.__name__
        ]
        dicts.append(dict(source))
        self._persist(dicts)
        return self._owned_entry(transformer_cls.__name__) or dicts[-1]

    def _reset_to_default(self, transformer_cls: type):
        dicts = [
            d
            for d in self.workflow.post_processors_dicts
            if d.get("name") != transformer_cls.__name__
        ]
        self._persist(dicts)
        self._rebuild()

    # ── Widget callbacks ──────────────────────────────────────────

    def _on_param_changed(
        self,
        group,
        *,
        key: str,
        value: Any,
        name: str,
    ):
        transformer_cls, owned, source = self._group_dicts[group]
        if owned is not None:
            self._save_param(owned, key, value)
            return

        # First interaction on a group that follows the machine
        # default (or bare defaults): claim an entry seeded from the
        # configuration the widget is displaying, then apply the
        # change to it. No rebuild — the widget already shows the
        # target state.
        entry = self._claim_entry(transformer_cls, source)
        self._group_dicts[group] = (transformer_cls, entry, source)
        self._save_param(entry, key, value)
