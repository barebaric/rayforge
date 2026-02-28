from typing import Set
from gettext import gettext as _
from gi.repository import Adw, Gtk


class MissingFeaturesDialog(Adw.MessageDialog):
    """
    Dialog shown when a document uses features that are not available.

    This happens when a document contains steps whose producer types
    are not registered (e.g., because the addon providing them is not
    installed).
    """

    def __init__(self, parent: Gtk.Window, missing_types: Set[str]):
        super().__init__(transient_for=parent, modal=True)
        self.set_heading(_("Missing Features"))

        if len(missing_types) == 1:
            msg = _(
                "This document uses a feature that is not available: {}"
            ).format(list(missing_types)[0])
        else:
            types_list = ", ".join(sorted(missing_types))
            msg = _(
                "This document uses features that are not available: {}"
            ).format(types_list)

        msg += "\n\n" + _("The document can still be edited and saved.")
        self.set_body(msg)

        self.add_response("ok", _("_OK"))
        self.set_default_response("ok")
        self.set_close_response("ok")
