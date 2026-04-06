import logging
from gettext import gettext as _
from gi.repository import Adw, Gtk
from ..addon_manager.addon_list import AddonListWidget
from ..shared.preferences_page import TrackedPreferencesPage

logger = logging.getLogger(__name__)


class AddonManagerPage(TrackedPreferencesPage):
    """
    Widget for managing installed addons.
    """

    key = "addons"

    def __init__(self):
        super().__init__(
            title=_("Addons"),
            icon_name="addon-symbolic",
        )

        # The list of addons, which is an Adw.PreferencesGroup
        self.addon_list_widget = AddonListWidget(
            title=_("Installed Addons"),
            description=_("Install, update, and remove addons."),
        )
        self.add(self.addon_list_widget)

        # The progress indicator must be wrapped in a PreferencesGroup
        # to be added to a PreferencesPage.
        progress_group = Adw.PreferencesGroup()

        self.progress_indicator = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_top=12,
            margin_bottom=12,
            halign=Gtk.Align.CENTER,
        )
        spinner = Gtk.Spinner()
        spinner.start()
        self.progress_label = Gtk.Label()
        self.progress_indicator.append(spinner)
        self.progress_indicator.append(self.progress_label)
        self.progress_indicator.set_visible(False)  # Hidden by default

        progress_group.add(self.progress_indicator)
        self.add(progress_group)

        # Connect signals to control the progress indicator
        self.addon_list_widget.install_started.connect(
            self._on_install_started
        )
        self.addon_list_widget.install_finished.connect(
            self._on_install_finished
        )

    def _on_install_started(self, sender, message: str):
        """Called when the list widget starts an installation."""
        self.progress_label.set_text(message)
        self.progress_indicator.set_visible(True)

    def _on_install_finished(self, sender):
        """Called when the installation is complete."""
        self.progress_indicator.set_visible(False)
