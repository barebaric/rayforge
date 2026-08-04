"""Step 2 — Choose controller.

Lists every available driver (GRBL, Ruida, Smoothieware,
OctoPrint, Marlin, plus the ``NoDeviceDriver`` affordance for
G-code-only export) as a grid of large icon buttons so the user
can pick the firmware / protocol family at a glance. There is no
default selection: the user must consciously choose a controller
(or "None") before the wizard will let them proceed.
"""

from gettext import gettext as _
from typing import Dict, List, Type

from blinker import Signal
from gi.repository import Gtk

from ....machine.device.profile import DeviceProfile
from ....machine.driver import drivers
from ....machine.driver.driver import Driver
from ...icons import get_icon
from . import WizardPage, _makePreferencesGroup

# Symbolic icon shown on each driver's tile. New drivers without an
# entry fall back to a generic device icon.
_DRIVER_ICONS: Dict[str, str] = {
    "GrblNetworkDriver": "network-wired-symbolic",
    "GrblTelnetDriver": "network-wired-symbolic",
    "RuidaDriver": "network-wired-symbolic",
    "GrblSerialDriver": "drive-removable-media-symbolic",
    "GrblSerialSimpleDriver": "drive-removable-media-symbolic",
    "MarlinSerialDriver": "drive-removable-media-symbolic",
    "OctoPrintDriver": "network-server-symbolic",
    "SmoothieDriver": "computer-symbolic",
}
_FALLBACK_ICON = "drive-harddisk-symbolic"
_EXPORT_ONLY_ICON = "document-save-symbolic"


class ControllerPage(WizardPage):
    step_number = 2
    title = _("Choose Controller")
    subtitle = _("What kind of controller board does this machine use?")

    def __init__(self, wizard, **kwargs):
        # Fired when the user picks a controller tile; the wizard
        # applies the choice and advances immediately.
        self.controller_selected = Signal()
        # Pre-compute the sorted, de-duplicated driver list before
        # build_ui() runs (the base __init__ calls build_ui last).
        driver_set: List[Type[Driver]] = []
        seen_classnames: set = set()
        for d in drivers:
            if d.__name__ in seen_classnames:
                continue
            # Hide the bare NoDeviceDriver from the controller list: it
            # is offered as an explicit tile below so the user-facing
            # label is friendlier.
            if d.__name__ == "NoDeviceDriver":
                continue
            driver_set.append(d)
            seen_classnames.add(d.__name__)

        self._drivers: List[Type[Driver]] = sorted(
            driver_set, key=lambda d: d.label.lower()
        )
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.group = _makePreferencesGroup(
            title=_("Controller"),
            description=_(
                "Pick the firmware / protocol family for this "
                "machine. If you aren't sure, choose the closest "
                "match — you can refine individual settings later."
            ),
        )
        self.content.append(self.group)

        self.flow_box = Gtk.FlowBox()
        self.flow_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.flow_box.set_homogeneous(True)
        self.flow_box.set_min_children_per_line(2)
        self.flow_box.set_max_children_per_line(4)
        self.flow_box.set_column_spacing(12)
        self.flow_box.set_row_spacing(12)
        self.flow_box.set_activate_on_single_click(True)
        self.flow_box.connect("child-activated", self._on_child_activated)
        self.content.append(self.flow_box)

        self._tiles: List[Gtk.FlowBoxChild] = []
        for index, d in enumerate(self._drivers):
            self._tiles.append(
                self._make_tile(
                    index,
                    d.label,
                    d.subtitle or "",
                    _DRIVER_ICONS.get(d.__name__, _FALLBACK_ICON),
                )
            )
        # Sentinel for None / export-only.
        self._tiles.append(
            self._make_tile(
                len(self._drivers),
                _("None — G-code export only"),
                _("No physical controller; export G-code to a file"),
                _EXPORT_ONLY_ICON,
            )
        )
        for tile in self._tiles:
            self.flow_box.append(tile)

        # No default selection: the user must consciously pick a
        # controller (or "None") before proceeding.
        self.set_ready(False)

    def _make_tile(
        self,
        index: int,
        title: str,
        subtitle: str,
        icon_name: str,
    ) -> Gtk.FlowBoxChild:
        child = Gtk.FlowBoxChild()
        button = Gtk.Button()
        button.add_css_class("flat")
        button.add_css_class("card")
        button.connect("clicked", self._on_tile_clicked, child)

        box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_top=16,
            margin_bottom=16,
            margin_start=12,
            margin_end=12,
        )
        image = get_icon(icon_name)
        image.set_pixel_size(40)
        title_label = Gtk.Label(
            label=title, wrap=True, justify=Gtk.Justification.CENTER
        )
        title_label.add_css_class("title-4")
        subtitle_label = Gtk.Label(
            label=subtitle, wrap=True, justify=Gtk.Justification.CENTER
        )
        subtitle_label.add_css_class("dim-label")
        box.append(image)
        box.append(title_label)
        box.append(subtitle_label)
        button.set_child(box)
        child.set_child(button)
        return child

    def enter(self, profile: DeviceProfile) -> None:
        """Re-select the tile matching the working profile's driver."""
        self.flow_box.unselect_all()
        driver_name = profile.machine_config.driver
        if driver_name:
            for index, d in enumerate(self._drivers):
                if d.__name__ == driver_name:
                    self.flow_box.select_child(self._tiles[index])
                    self.set_ready(True)
                    return
        self.set_ready(False)

    # ----- selection -----------------------------------------------------

    def _on_tile_clicked(
        self, button: Gtk.Button, child: Gtk.FlowBoxChild
    ) -> None:
        self._select_child(child)

    def _on_child_activated(
        self, flow_box: Gtk.FlowBox, child: Gtk.FlowBoxChild
    ) -> None:
        self._select_child(child)

    def _select_child(self, child: Gtk.FlowBoxChild) -> None:
        self.flow_box.select_child(child)
        self.set_ready(True)
        index = self._tiles.index(child)
        driver_name: str | None
        if index < len(self._drivers):
            driver_name = self._drivers[index].__name__
        else:
            driver_name = None
        self.controller_selected.send(self, driver=driver_name)

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        selected = self.flow_box.get_selected_children()
        if not selected:
            return False
        index = self._tiles.index(selected[0])
        if index < len(self._drivers):
            profile.machine_config.driver = self._drivers[index].__name__
        else:
            profile.machine_config.driver = None
        return True


__all__ = ["ControllerPage"]
