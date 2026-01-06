import logging
from gi.repository import Adw, Gtk
from ..package_manager.package_list import PackageListWidget

logger = logging.getLogger(__name__)


class PackageManagerPage(Adw.PreferencesPage):
    """
    Widget for managing installed packages.
    """

    def __init__(self):
        super().__init__(
            title=_("Packages"),
            icon_name="addon-symbolic",
        )

        # The list of packages, which is an Adw.PreferencesGroup
        self.package_list_widget = PackageListWidget(
            title=_("Installed Packages"),
            description=_("Install, update, and remove packages and plugins."),
        )
        self.add(self.package_list_widget)

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
        self.package_list_widget.install_started.connect(
            self._on_install_started
        )
        self.package_list_widget.install_finished.connect(
            self._on_install_finished
        )

    def _on_install_started(self, sender, message: str):
        """Called when the list widget starts an installation."""
        self.progress_label.set_text(message)
        self.progress_indicator.set_visible(True)

    def _on_install_finished(self, sender):
        """Called when the installation is complete."""
        self.progress_indicator.set_visible(False)
