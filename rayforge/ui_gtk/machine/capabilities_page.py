from gettext import gettext as _
from typing import List

from gi.repository import Adw

from ...core.capability import MachineCapability
from ...machine.models.laser import LaserHead
from ...machine.models.machine import Machine
from ...machine.models.spindle import SpindleHead
from ..shared.preferences_page import TrackedPreferencesPage


class CapabilitiesPage(TrackedPreferencesPage):
    """Machine settings page showing the machine's capabilities."""

    key = "capabilities"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Capabilities"),
            icon_name="settings-symbolic",
            **kwargs,
        )
        self.machine = machine
        self._rows: List[Adw.ActionRow] = []

        self.capability_group = Adw.PreferencesGroup(
            title=_("Machine Capabilities"),
            description=_(
                "Capabilities are inferred from the machine's heads and "
                "any explicit configuration. They control which steps are "
                "offered when adding to a workflow."
            ),
        )
        self.add(self.capability_group)

        self.machine.changed.connect(self._refresh)
        self._refresh()

        self.connect("destroy", self._on_destroy)

    def _source_text(self, capability: MachineCapability) -> str:
        """Describes where a capability comes from."""
        sources = []
        for head in self.machine.heads:
            if head.machine_capability == capability:
                if isinstance(head, LaserHead):
                    sources.append(_("Laser Head"))
                elif isinstance(head, SpindleHead):
                    sources.append(_("Spindle Head"))
                else:
                    sources.append(head.name)
        if self.machine._explicit_capabilities and (
            capability in self.machine._explicit_capabilities
        ):
            sources.append(_("explicit configuration"))
        if not sources:
            return _("unknown source")
        return ", ".join(sources)

    def _refresh(self, sender=None, **kwargs):
        """Rebuilds the capability list from the machine."""
        for row in self._rows:
            self.capability_group.remove(row)
        self._rows.clear()

        for cap in sorted(
            self.machine.get_capabilities(), key=lambda c: c.value
        ):
            row = Adw.ActionRow(
                title=cap.label,
                subtitle=f"{cap.description} · {self._source_text(cap)}",
            )
            self.capability_group.add(row)
            self._rows.append(row)

    def _on_destroy(self, *args):
        """Disconnects signals to prevent memory leaks."""
        self.machine.changed.disconnect(self._refresh)
