"""Dialog for confirming the installation of a community addon."""

import webbrowser
from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw, GLib, Gtk, Pango


class CommunityAddonDialog(Adw.MessageDialog):
    """
    Confirmation dialog shown before installing a community addon.

    Community addons are not provided by the Rayforge team, so the
    user is asked to only continue if they trust the addon's author.
    The dialog links to the addon repository so that its code can be
    reviewed before installing.
    """

    def __init__(
        self,
        addon_name: str,
        repository_url: str | None = None,
        author: str | None = None,
        on_install: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ):
        super().__init__()

        self._on_install = on_install
        self._on_cancel = on_cancel

        self.set_heading(_("Install Community Addon?"))
        self.set_body(
            _(
                '"{name}" is a community addon and is not provided by '
                "the Rayforge team. Addons have the same access to "
                "your computer and your files as Rayforge itself. "
                "Only install this addon if you trust its author, "
                "{author}."
            ).format(name=addon_name, author=author or _("unknown"))
        )

        if repository_url:
            self.set_extra_child(self._build_repository_link(repository_url))

        self.add_response("cancel", _("Cancel"))
        self.add_response("install", _("Install Anyway"))
        self.set_response_appearance(
            "install", Adw.ResponseAppearance.DESTRUCTIVE
        )
        self.set_default_response("cancel")
        self.set_close_response("cancel")

        # libadwaita stacks the response buttons vertically whenever
        # their combined width exceeds the dialog width, so give the
        # dialog enough width to keep them on one row.
        self.set_default_size(480, -1)

        self.connect("response", self._on_response)

    def _build_repository_link(self, repository_url: str) -> Gtk.Label:
        """
        Create a clickable link for reviewing the addon's code.

        The URL itself is used as the link text so that the user can
        see where the link leads before clicking it.
        """
        label = Gtk.Label(
            use_markup=True,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
            max_width_chars=40,
            margin_top=12,
        )
        label.set_markup(
            '<a href="{url}" title="{tooltip}">{text}</a>'.format(
                url=GLib.markup_escape_text(repository_url),
                tooltip=GLib.markup_escape_text(
                    _("Open the repository in your browser")
                ),
                text=GLib.markup_escape_text(repository_url),
            )
        )
        label.connect("activate-link", self._on_activate_link)
        return label

    def _on_activate_link(self, label, uri: str) -> bool:
        webbrowser.open(uri)
        return True

    def _on_response(self, dialog, response_id: str):
        if response_id == "install":
            if self._on_install:
                self._on_install()
        elif self._on_cancel:
            self._on_cancel()
        if self.get_root() is not None:
            self.close()
