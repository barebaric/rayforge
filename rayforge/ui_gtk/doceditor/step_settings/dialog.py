from gettext import gettext as _
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from gi.repository import Adw, Gtk

from rayforge.context import get_context
from rayforge.core.step import Step
from rayforge.ui_gtk.doceditor.step_settings.pages import (
    GeneralStepSettingsPage,
    PostProcessingPage,
    StepSettingsPage,
)
from rayforge.ui_gtk.icons import get_icon
from rayforge.ui_gtk.shared.patched_dialog_window import PatchedDialogWindow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class StepSettingsDialog(PatchedDialogWindow):
    _open_dialogs: Dict[int, "StepSettingsDialog"] = {}

    def __init__(
        self,
        editor: "DocEditor",
        step: Step,
        **kwargs,
    ):
        super().__init__(skip_usage_tracking=True, **kwargs)
        self.editor = editor
        self.step = step
        self.set_title(_("{name} Settings").format(name=step.name))

        # Adw.ToolbarView provides areas for a header, content, and bottom bar.
        main_view = Adw.ToolbarView()
        self.set_content(main_view)

        # A HeaderBar provides the window decorations (close button, etc.)
        header = Adw.HeaderBar()
        main_view.add_top_bar(header)

        # Gtk.Stack holds the pages.
        self.stack = Gtk.Stack()
        main_view.set_content(self.stack)

        # Set a reasonable default size to avoid being too narrow
        self.set_default_size(600, 750)

        # Destroy window on close to prevent leaks
        self.set_hide_on_close(False)
        self.connect("close-request", self._on_close_request)

        # --- Main Step Settings + addon-provided extra pages ---
        context = get_context()
        self.general_view: Optional[StepSettingsPage] = None
        self._extra_pages: List[
            Tuple[str, StepSettingsPage, Optional[str]]
        ] = []
        if context:
            context.plugin_mgr.hook.step_settings_loaded(
                dialog=self, step=self.step, producer=None
            )
        if self.general_view is None:
            self.general_view = GeneralStepSettingsPage(self.editor, self.step)
        scrolled_page1 = Gtk.ScrolledWindow(
            child=self.general_view,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        self.stack.add_named(scrolled_page1, "step-settings")

        self._extra_page_names = []
        for index, (title, page, icon_name) in enumerate(
            self._extra_pages, start=1
        ):
            scrolled = Gtk.ScrolledWindow(
                child=page,
                hscrollbar_policy=Gtk.PolicyType.NEVER,
                vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            )
            page_name = f"settings-{index}"
            self.stack.add_named(scrolled, page_name)
            self._extra_page_names.append(page_name)

        # --- Post Processing Settings ---
        self.post_processing_view = PostProcessingPage(self.editor, self.step)
        scrolled_page2 = Gtk.ScrolledWindow(
            child=self.post_processing_view,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
        )
        self.stack.add_named(scrolled_page2, "post-processing")

        # --- Build the custom switcher ---
        switcher_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        switcher_box.add_css_class("linked")
        header.set_title_widget(switcher_box)

        self.btn_step_settings = Gtk.ToggleButton()
        self.btn_step_settings.set_child(
            self._create_tab_title(_("Step Settings"), "laser-path-symbolic")
        )
        self.btn_step_settings.connect(
            "toggled", self._on_tab_toggled, self.stack, "step-settings"
        )
        switcher_box.append(self.btn_step_settings)

        self._extra_buttons = []
        for index, (title, page, icon_name) in enumerate(
            self._extra_pages, start=1
        ):
            button = Gtk.ToggleButton(group=self.btn_step_settings)
            button.set_child(
                self._create_tab_title(title, icon_name or "settings-symbolic")
            )
            button.connect(
                "toggled",
                self._on_tab_toggled,
                self.stack,
                f"settings-{index}",
            )
            switcher_box.append(button)
            self._extra_buttons.append(button)

        self.btn_post_processing = Gtk.ToggleButton(
            group=self.btn_step_settings
        )
        self.btn_post_processing.set_child(
            self._create_tab_title(
                _("Post Processing"), "post-processor-symbolic"
            )
        )
        self.btn_post_processing.connect(
            "toggled", self._on_tab_toggled, self.stack, "post-processing"
        )
        switcher_box.append(self.btn_post_processing)

        has_post_processors = bool(
            step.per_step_transformers_dicts
            or step.per_workpiece_transformers_dicts
        )
        self.btn_post_processing.set_visible(has_post_processors)

        # Default to step-settings page
        self.btn_step_settings.set_active(True)
        if self.general_view is not None:
            self.general_view._sync_widgets_to_model()

    def set_step_settings_page(self, page: StepSettingsPage):
        """Provide the step's settings page (called by addon hooks)."""
        self.general_view = page

    def add_settings_page(
        self,
        title: str,
        page: StepSettingsPage,
        icon_name: Optional[str] = None,
    ):
        """Provide an additional settings page (called by addon hooks)."""
        self._extra_pages.append((title, page, icon_name))

    @classmethod
    def present_for_step(
        cls,
        editor: "DocEditor",
        step: Step,
        parent_window: Optional[Gtk.Root],
    ) -> "StepSettingsDialog":
        existing = cls._open_dialogs.get(id(step))
        if existing:
            existing.present()
            return existing
        dialog = cls(editor, step, transient_for=parent_window)
        cls._open_dialogs[id(step)] = dialog
        dialog.connect("close-request", cls._on_dialog_closed)
        dialog.present()
        return dialog

    @classmethod
    def _on_dialog_closed(cls, dialog: "StepSettingsDialog", *args) -> bool:
        cls._open_dialogs.pop(id(dialog.step), None)
        return False

    def set_initial_page(self, page: str):
        """Set the initial visible page after dialog construction."""
        if page == "post-processing":
            self.btn_post_processing.set_active(True)
            return
        for index, (page_title, extra_page, icon_name) in enumerate(
            self._extra_pages
        ):
            if (
                page.lower() == page_title.lower()
                or page == f"settings-{index + 1}"
            ):
                self._extra_buttons[index].set_active(True)
                return
        self.btn_step_settings.set_active(True)

    def _on_tab_toggled(self, button, stack, page_name):
        """Callback to switch the Gtk.Stack page."""
        if button.get_active():
            stack.set_visible_child_name(page_name)

    def _create_tab_title(self, title_str: str, icon_name: str) -> Gtk.Widget:
        """Creates a box with an icon and a label for a tab button."""
        icon = get_icon(icon_name)
        label = Gtk.Label(label=title_str)
        box = Gtk.Box(spacing=6, orientation=Gtk.Orientation.HORIZONTAL)
        box.append(icon)
        box.append(label)
        return box

    def _on_close_request(self, window):
        # Clean up debounce timers in all settings pages to prevent GLib
        # warnings when the window is closed.
        pages = (
            [self.general_view]
            + [page for _, page, _ in self._extra_pages]
            + [self.post_processing_view]
        )
        for page in pages:
            cleanup = getattr(page, "_cleanup", None)
            if callable(cleanup):
                cleanup()
        return False  # Allow the window to close
