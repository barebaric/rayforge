"""Base class for camera-wizard pages."""

from typing import TYPE_CHECKING, List, Optional

from gi.repository import Gtk

from ....camera.controller import CameraController

if TYPE_CHECKING:
    from .wizard import CameraWizard


class CameraWizardPage:
    """A single step of the camera calibration wizard.

    Pages own a region of the wizard's stack and a set of footer
    buttons. The wizard drives them via :meth:`enter` (shown) and
    reads :meth:`can_proceed` / :meth:`footer_buttons` to update the
    footer. Pages mutate the shared :class:`CameraController` /
    ``Camera`` model directly; no separate apply step is needed.

    Flow transitions (e.g. a branch chosen, a step completed) are
    signalled via :class:`blinker.Signal`s the wizard connects to,
    so pages never call wizard methods directly. Ambient UI
    affordances (toasts, error dialogs) are exposed here as methods
    the wizard provides, keeping pages decoupled from the dialog
    shell.
    """

    step_name: str = ""
    title: str = ""

    def __init__(self, wizard: "CameraWizard", controller: CameraController):
        self.wizard = wizard
        self.controller = controller
        self.root: Optional[Gtk.Box] = None

    def build(self) -> Gtk.Box:
        raise NotImplementedError

    def enter(self) -> None:
        pass

    def leave(self) -> None:
        pass

    def can_proceed(self) -> bool:
        return True

    def footer_buttons(self) -> List[Gtk.Button]:
        return []

    def back_target(self) -> Optional[str]:
        return None

    # ----- ambient UI affordances (provided by the wizard) ------------

    def show_toast(self, message: str) -> None:
        self.wizard.show_toast(message)

    def show_error(self, title: str, message: str) -> None:
        self.wizard.show_error(title, message)


__all__ = ["CameraWizardPage"]
