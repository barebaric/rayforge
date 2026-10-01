from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...context import get_context
from ...usage import get_usage_tracker


class UsageConsentDialog(Adw.MessageDialog):
    def __init__(self, parent: Gtk.Window):
        super().__init__(transient_for=parent, modal=True)
        self.set_heading(_("Help me prioritize what to build"))
        self.set_body(
            _(
                "Anonymous usage data shows me which machines and "
                "features people actually use. It never includes "
                "project content or personal data."
            )
        )

        link_label = Gtk.Label(
            label=_(
                '<a href="https://rayforge.org/docs/general-info/'
                'usage-tracking">Learn more</a> about usage tracking '
                "and privacy."
            ),
            use_markup=True,
            wrap=True,
            max_width_chars=50,
            halign=Gtk.Align.START,
            margin_top=12,
        )
        self.set_extra_child(link_label)

        self.add_response("decline", _("No Thanks"))
        self.add_response("accept", _("Allow Reporting"))
        self.set_response_appearance(
            "accept", Adw.ResponseAppearance.SUGGESTED
        )
        self.set_default_response("accept")
        self.set_close_response("decline")

        self.connect("response", self._on_response)

    def _on_response(self, dialog, response_id):
        consent = response_id == "accept"
        context = get_context()
        context.config.set_usage_consent(consent)
        get_usage_tracker().set_enabled(consent)
        self.close()
