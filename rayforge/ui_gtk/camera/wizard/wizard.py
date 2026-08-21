"""Unified camera configuration wizard.

Walks the full camera setup sub-workflow:

* image settings (resolution, white balance, brightness, ...)
* lens calibration (choice of automatic Charuco capture or manual
  coefficient entry, or skip)
* world alignment

The wizard is launched from the machine wizard's camera page and from
the camera preferences page's "Wizard" button.
"""

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ....camera.controller import CameraController
from ...shared.patched_dialog_window import PatchedDialogWindow
from .alignment_page import AlignmentPage
from .base_page import CameraWizardPage
from .capture_page import CapturePage
from .card_page import CardPage
from .image_settings_page import ImageSettingsPage
from .lens_calibration_choice_page import LensCalibrationChoicePage
from .lens_calibration_settings_page import LensCalibrationSettingsPage

logger = logging.getLogger(__name__)


class CameraWizard(PatchedDialogWindow):
    """Page-based camera calibration wizard."""

    def __init__(
        self,
        parent: Gtk.Window,
        controller: CameraController,
        **kwargs,
    ):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1150,
            default_height=750,
            title=_("{camera} - Camera Wizard").format(
                camera=controller.config.name
            ),
            **kwargs,
        )
        self.controller = controller

        self._pages: dict[str, CameraWizardPage] = {}
        self._page_order: list[str] = []
        self._current: str | None = None
        self._history: list[str] = []

        self._setup_ui()
        self._build_pages()
        self._navigate_to(self._page_order[0], record_history=False)

    def _setup_ui(self) -> None:
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        header = Adw.HeaderBar()
        content.append(header)

        self._main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=32,
            margin_end=32,
        )
        content.append(self._main_box)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self._main_box.append(self._stack)

        self._button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_top=12,
        )
        self._main_box.append(self._button_box)

        self._back_btn = Gtk.Button(label=_("Back"))
        self._back_btn.add_css_class("flat")
        self._back_btn.connect("clicked", self._on_back_clicked)
        self._button_box.append(self._back_btn)

        self._cancel_btn = Gtk.Button(label=_("Cancel"))
        self._cancel_btn.add_css_class("flat")
        self._cancel_btn.connect("clicked", lambda _: self.close())
        self._button_box.append(self._cancel_btn)

        self._action_slot = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=12
        )
        self._button_box.append(self._action_slot)

        self._next_btn = Gtk.Button(label=_("Next"))
        self._next_btn.add_css_class("suggested-action")
        self._next_btn.connect("clicked", self._on_next_clicked)
        self._button_box.append(self._next_btn)

        self._finish_btn = Gtk.Button(label=_("Finish"))
        self._finish_btn.add_css_class("suggested-action")
        self._finish_btn.connect("clicked", lambda _: self.close())
        self._button_box.append(self._finish_btn)

        self._stack.connect("notify::visible-child", self._on_page_changed)

    def _build_pages(self) -> None:
        # Fixed prefix: image settings, then the lens-branch choice.
        self._register(ImageSettingsPage)
        self._register(LensCalibrationChoicePage)
        # Branch pages are built up front so they are ready when the
        # user picks a branch, but not added to the order until then.
        self._register(CardPage, add_to_order=False)
        self._register(CapturePage, add_to_order=False)
        self._register(LensCalibrationSettingsPage, add_to_order=False)
        self._register(AlignmentPage, add_to_order=False)

        for name in self._page_order:
            self._stack.add_named(self._pages[name].build(), name)

    def _register(self, cls, *, add_to_order: bool = True) -> CameraWizardPage:
        page = cls(self, self.controller)
        self._pages[page.step_name] = page
        if add_to_order:
            self._page_order.append(page.step_name)
        self._wire_page_signals(page)
        return page

    def _wire_page_signals(self, page: CameraWizardPage) -> None:
        if isinstance(page, LensCalibrationChoicePage):
            page.branch_chosen.connect(self._on_lens_branch_chosen)
        elif isinstance(page, AlignmentPage):
            page.alignment_applied.connect(self._on_alignment_applied)

    def _on_lens_branch_chosen(self, _sender, **kwargs) -> None:
        self.on_lens_branch_chosen(kwargs["branch"])

    def _on_alignment_applied(self, _sender) -> None:
        self.on_alignment_applied()

    def _ensure_in_stack(self, name: str) -> None:
        page = self._pages[name]
        if self._stack.get_child_by_name(name) is None:
            self._stack.add_named(
                page.root if page.root is not None else page.build(),
                name,
            )

    # ----- branch handling ---------------------------------------------

    def on_lens_branch_chosen(self, branch: str) -> None:
        """Insert the branch-specific pages after the choice page.

        All branches land on the alignment page (the terminal step);
        Skip just omits the lens-calibration pages.
        """
        choice_idx = self._page_order.index("lens_choice")
        # Truncate the order back to the choice page so a re-pick
        # rebuilds the tail cleanly.
        self._page_order = self._page_order[: choice_idx + 1]

        if branch == LensCalibrationChoicePage.BRANCH_AUTOMATIC:
            self._append_branch_page("card")
            self._append_branch_page("capture")
        elif branch == LensCalibrationChoicePage.BRANCH_MANUAL:
            self._append_branch_page("lens_manual")
        # Skipped: no lens-calibration pages.

        # Image↔world alignment is always the terminal page (Finish).
        self._append_branch_page("alignment")

        # Navigate to the first page after the choice.
        self._navigate_to(self._page_order[choice_idx + 1])

    def on_alignment_applied(self) -> None:
        """The user applied the alignment — finish the wizard."""
        self.close()

    def _append_branch_page(self, name: str) -> None:
        self._ensure_in_stack(name)
        self._page_order.append(name)

    # ----- navigation ---------------------------------------------------

    def _navigate_to(self, name: str, *, record_history: bool = True) -> None:
        if record_history and self._current is not None:
            self._history.append(self._current)
        if self._current is not None and self._current != name:
            self._pages[self._current].leave()
        # Slide forward when advancing, backward when returning.
        if self._current is not None and name in self._page_order:
            cur_idx = (
                self._page_order.index(self._current)
                if self._current in self._page_order
                else -1
            )
            new_idx = self._page_order.index(name)
            if new_idx >= cur_idx:
                self._stack.set_transition_type(
                    Gtk.StackTransitionType.SLIDE_LEFT
                )
            else:
                self._stack.set_transition_type(
                    Gtk.StackTransitionType.SLIDE_RIGHT
                )
        self._current = name
        page = self._pages[name]
        if isinstance(page, CapturePage):
            self._sync_capture_board(page)
        page.enter()
        self._stack.set_visible_child_name(name)
        self._update_footer(name, page)

    def _sync_capture_board(self, capture: CapturePage) -> None:
        """Propagate the calibration card board to the capture page.

        The card page owns the generated :class:`CharucoBoard` (it may
        be regenerated when the user tweaks the card size). Without this
        the capture page never receives a board, so its calibrator is
        never initialised and Capture Frame silently does nothing.
        """
        card = self._pages.get("card")
        if not isinstance(card, CardPage):
            return
        if card.board is not None:
            capture.set_board(card.board)

    def _on_page_changed(self, _stack, _pspec) -> None:
        name = self._stack.get_visible_child_name()
        if name is None or name == self._current:
            return
        self._navigate_to(name)

    def _update_footer(self, name: str, page: CameraWizardPage) -> None:
        page_title = page.title or _("Camera Wizard")
        self.set_title(f"{page_title} — {self.controller.config.name}")
        idx = self._page_order.index(name)
        self._back_btn.set_visible(idx > 0)
        # Finish only on the terminal alignment page; the lens-choice
        # page is a decision point, not a terminal.
        is_terminal = name == "alignment"
        self._finish_btn.set_visible(is_terminal)
        self._next_btn.set_visible(not is_terminal)
        self._next_btn.set_sensitive(page.can_proceed())

        child = self._action_slot.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._action_slot.remove(child)
            child = nxt
        for btn in page.footer_buttons():
            self._action_slot.append(btn)

    def _on_back_clicked(self, _btn) -> None:
        if self._current is None:
            return
        idx = self._page_order.index(self._current)
        if idx <= 0:
            return
        self._navigate_to(self._page_order[idx - 1])

    def _on_next_clicked(self, _btn) -> None:
        if self._current is None:
            return
        idx = self._page_order.index(self._current)
        if idx + 1 >= len(self._page_order):
            return
        self._navigate_to(self._page_order[idx + 1])

    # ----- helpers ------------------------------------------------------

    def show_error(self, title: str, message: str) -> None:
        dialog = Adw.MessageDialog(
            transient_for=self,
            modal=True,
            heading=title,
            body=message,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present()

    def show_toast(self, message: str) -> None:
        self.toast_overlay.add_toast(Adw.Toast.new(message))

    def close(self):
        for page in self._pages.values():
            if isinstance(page, CapturePage):
                page.stop()
            elif isinstance(page, (ImageSettingsPage, AlignmentPage)):
                page.leave()
        super().close()


__all__ = ["CameraWizard"]
