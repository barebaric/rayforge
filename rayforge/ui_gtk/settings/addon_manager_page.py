import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...context import get_context
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

        context = get_context()
        services_group = Adw.PreferencesGroup(
            title=_("Online Services"),
            description=_(
                "Controls registry access, remote installation, and "
                "external license services. Built-in and locally installed "
                "addons remain available when disabled."
            ),
        )
        self.external_services_row = Adw.SwitchRow(
            title=_("Enable external addon services"),
            subtitle=_("Allow network access to addon and license providers"),
        )
        self.external_services_row.set_active(
            context.config.external_addon_services_enabled
        )
        self.external_services_row.connect(
            "notify::active", self._on_external_services_changed
        )
        services_group.add(self.external_services_row)
        self.add(services_group)

        # The list of addons, which is an Adw.PreferencesGroup
        self.addon_list_widget = AddonListWidget(
            title=_("Installed Addons"),
            description=_("Install, update, and remove addons."),
        )
        self.addon_list_widget.set_external_services_enabled(
            context.config.external_addon_services_enabled
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

    def _on_external_services_changed(self, switch_row, _):
        enabled = switch_row.get_active()
        context = get_context()
        context.config.set_external_addon_services_enabled(enabled)
        context.license_validator.invalidate_cache()
        context.license_validator.changed.send(context.license_validator)
        self.addon_list_widget.set_external_services_enabled(enabled)

    def _on_install_started(self, sender, message: str):
        """Called when the list widget starts an installation."""
        self.progress_label.set_text(message)
        self.progress_indicator.set_visible(True)

    def _on_install_finished(self, sender):
        """Called when the installation is complete."""
        self.progress_indicator.set_visible(False)
