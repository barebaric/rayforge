"""Full-size view of an addon screenshot."""

from gettext import gettext as _

from gi.repository import Adw, Gdk, Gtk

from ..shared.patched_dialog_window import PatchedDialogWindow


class ScreenshotViewDialog(PatchedDialogWindow):
    """
    Shows a screenshot at a larger size.

    Opens when the user clicks a screenshot in the addon detail view.
    Clicking the image or pressing Escape closes it.
    """

    def __init__(self, parent_window, texture: Gdk.Texture):
        super().__init__()
        self.set_transient_for(parent_window)
        self.set_modal(True)
        self.set_title(_("Screenshot"))
        self.set_default_size(
            min(texture.get_width() + 48, 1400),
            min(texture.get_height() + 96, 900),
        )

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(box)

        header = Adw.HeaderBar()
        box.append(header)

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_vexpand(True)
        scrolled.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        box.append(scrolled)

        picture = Gtk.Picture()
        picture.set_paintable(texture)
        picture.set_can_shrink(True)
        picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        scrolled.set_child(picture)

        click = Gtk.GestureClick()
        click.connect("released", lambda *_args: self.close())
        picture.add_controller(click)
