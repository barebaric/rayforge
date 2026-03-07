"""AI settings page with inline editor for Rayforge."""

import asyncio
import logging
import uuid
from gettext import gettext as _
from typing import Optional, cast

from gi.repository import Adw, GLib, Gtk
from blinker import Signal

from ...context import get_context
from ...core.ai.provider import AIProviderConfig, AIProviderType
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from ..shared.preferences_page import TrackedPreferencesPage

logger = logging.getLogger(__name__)


class ProviderRow(Gtk.Box):
    """A widget representing a single AI provider in a ListBox."""

    def __init__(
        self,
        provider_id: str,
        config: AIProviderConfig,
        is_default: bool,
        on_toggle_enabled,
        on_delete_callback,
        on_set_default_callback,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.provider_id = provider_id
        self.config = config
        self.is_default = is_default
        self.on_toggle_enabled = on_toggle_enabled
        self.on_delete_callback = on_delete_callback
        self.on_set_default_callback = on_set_default_callback
        self._setup_ui()

    def _setup_ui(self):
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        self.default_icon = get_icon("check-circle-symbolic")
        self.default_icon.set_tooltip_text(_("Default provider"))
        self.default_icon.set_valign(Gtk.Align.CENTER)
        self.default_icon.set_visible(self.is_default)
        self.append(self.default_icon)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        self.title_label = Gtk.Label(
            label=self.config.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        self._update_title_style()
        labels_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(
            label=self.config.provider_type.value.replace("_", " ").title(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        self.subtitle_label.add_css_class("dim-label")
        self.subtitle_label.add_css_class("caption")
        labels_box.append(self.subtitle_label)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        self.enable_switch = Gtk.Switch(valign=Gtk.Align.CENTER)
        self.enable_switch.set_active(self.config.enabled)
        self.enable_switch.set_tooltip_text(
            _("Enable or disable this provider")
        )
        self.enable_switch.connect("state-set", self._on_toggle_clicked)
        suffix_box.append(self.enable_switch)

        self.default_btn = Gtk.Button(child=get_icon("check-symbolic"))
        self.default_btn.add_css_class("flat")
        self.default_btn.set_tooltip_text(_("Set as default"))
        self.default_btn.set_visible(not self.is_default)
        self.default_btn.connect("clicked", self._on_set_default_clicked)
        suffix_box.append(self.default_btn)

        delete_btn = Gtk.Button(child=get_icon("delete-symbolic"))
        delete_btn.add_css_class("flat")
        delete_btn.connect("clicked", self._on_delete_clicked)
        suffix_box.append(delete_btn)

    def update_from_config(self, config: AIProviderConfig, is_default: bool):
        self.config = config
        self.is_default = is_default
        self.title_label.set_label(config.name)
        self._update_title_style()
        self.default_icon.set_visible(is_default)
        self.default_btn.set_visible(not is_default)
        self.enable_switch.set_active(config.enabled)

    def _update_title_style(self):
        if self.config.enabled:
            self.title_label.remove_css_class("dim-label")
        else:
            self.title_label.add_css_class("dim-label")

    def _on_toggle_clicked(self, switch: Gtk.Switch, state: bool) -> bool:
        self.on_toggle_enabled(self.provider_id, state)
        return False

    def _on_delete_clicked(self, button):
        self.on_delete_callback(self.provider_id, self.config.name)

    def _on_set_default_clicked(self, button):
        self.on_set_default_callback(self.provider_id)


class ProviderListWidget(PreferencesGroupWithButton):
    """Widget for displaying and managing a list of AI providers."""

    def __init__(self, **kwargs):
        super().__init__(
            button_label=_("Add Provider"),
            selection_mode=Gtk.SelectionMode.SINGLE,
            **kwargs,
        )
        self.provider_selected = Signal()
        self._row_widgets: dict[str, ProviderRow] = {}
        self._setup_ui()

    def _setup_ui(self):
        placeholder = Gtk.Label(
            label=_("No providers configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.connect("row-selected", self._on_provider_selected)

    def populate_and_select(self, select_id: Optional[str] = None):
        ai_service = get_context().ai_service
        providers = list(ai_service.providers.items())
        default_id = ai_service.default_provider_id

        selected_row = self.list_box.get_selected_row()
        selected_provider_id = None
        if selected_row:
            child = selected_row.get_child()
            if isinstance(child, ProviderRow):
                selected_provider_id = child.provider_id

        row_count = 0
        while self.list_box.get_row_at_index(row_count):
            row_count += 1

        new_selection_index = -1
        for i, (provider_id, config) in enumerate(providers):
            if provider_id == selected_provider_id:
                new_selection_index = i

            if i < row_count:
                row = self.list_box.get_row_at_index(i)
                if row:
                    provider_row = cast(ProviderRow, row.get_child())
                    provider_row.update_from_config(
                        config, provider_id == default_id
                    )
                    self._row_widgets[provider_id] = provider_row
            else:
                list_box_row = Gtk.ListBoxRow()
                row_widget = self.create_row_widget((provider_id, config))
                list_box_row.set_child(row_widget)
                self.list_box.append(list_box_row)
                self._row_widgets[provider_id] = row_widget

        while row_count > len(providers):
            last_row = self.list_box.get_row_at_index(row_count - 1)
            if last_row:
                self.list_box.remove(last_row)
            row_count -= 1

        if new_selection_index >= 0:
            row = self.list_box.get_row_at_index(new_selection_index)
            if row:
                self.list_box.select_row(row)
        elif len(providers) > 0:
            row = self.list_box.get_row_at_index(0)
            if row:
                self.list_box.select_row(row)
        else:
            if self.list_box.get_selected_row():
                self.list_box.unselect_all()
            else:
                self._on_provider_selected(self.list_box, None)

    def get_row_for_provider(self, provider_id: str) -> Optional[ProviderRow]:
        return self._row_widgets.get(provider_id)

    def create_row_widget(self, item) -> ProviderRow:
        provider_id, config = item
        row_widget = ProviderRow(
            provider_id,
            config,
            provider_id == get_context().ai_service.default_provider_id,
            self._on_toggle_enabled,
            self._on_delete_provider,
            self._on_set_default,
        )
        return row_widget

    def _on_add_clicked(self, button: Gtk.Button):
        ai_service = get_context().ai_service
        new_config = AIProviderConfig(
            id=str(uuid.uuid4())[:8],
            name=_("New Provider"),
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            api_key="",
            base_url="https://api.openai.com/v1",
            default_model="",
            enabled=True,
        )
        ai_service.add_provider(new_config)

    def _on_toggle_enabled(self, provider_id: str, enabled: bool):
        ai_service = get_context().ai_service
        config = ai_service.get_config(provider_id)
        if config:
            new_config = AIProviderConfig(
                id=config.id,
                name=config.name,
                provider_type=config.provider_type,
                api_key=config.api_key,
                base_url=config.base_url,
                default_model=config.default_model,
                enabled=enabled,
            )
            ai_service.update_provider(new_config)

    def _on_delete_provider(self, provider_id: str, name: str):
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, root) if root else None,
            heading=_("Delete '{name}'?").format(name=name),
            body=_(
                "This AI provider will be permanently removed. "
                "This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")

        def on_response(d, response_id):
            if response_id == "delete":
                get_context().ai_service.remove_provider(provider_id)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_set_default(self, provider_id: str):
        get_context().ai_service.default_provider_id = provider_id
        self.populate_and_select(select_id=provider_id)

    def _on_provider_selected(
        self, listbox: Gtk.ListBox, row: Optional[Gtk.ListBoxRow]
    ):
        provider_id = None
        config = None
        selected_row = listbox.get_selected_row()

        if selected_row:
            child = selected_row.get_child()
            if isinstance(child, ProviderRow):
                provider_id = child.provider_id
                config = child.config

        self.provider_selected.send(
            self, provider_id=provider_id, config=config
        )


class ProviderEditorWidget(Adw.PreferencesGroup):
    """Inline editor widget for AI provider settings with instant apply."""

    def __init__(self, list_widget: "ProviderListWidget", **kwargs):
        super().__init__(**kwargs)
        self.list_widget = list_widget
        self.provider_id: Optional[str] = None
        self._updating = False
        self._setup_ui()

    def _setup_ui(self):
        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.connect("changed", self._on_name_changed)
        self.add(self.name_row)

        self.type_row = Adw.ComboRow(
            title=_("Provider Type"),
            model=Gtk.StringList.new([_("OpenAI Compatible")]),
        )
        self.type_row.set_selected(0)
        self.type_row.connect("notify::selected", self._on_field_changed)
        self.add(self.type_row)

        self.base_url_row = Adw.EntryRow(title=_("Base URL"))
        self.base_url_row.connect("changed", self._on_field_changed)
        self.add(self.base_url_row)

        self.api_key_row = Adw.PasswordEntryRow(title=_("API Key"))
        self.api_key_row.connect("changed", self._on_field_changed)
        self.add(self.api_key_row)

        self.model_row = Adw.EntryRow(title=_("Default Model"))
        self.model_row.connect("changed", self._on_field_changed)
        self.add(self.model_row)

        self.test_row = Adw.ActionRow(
            title=_("Connection Test"),
            subtitle=_("Verify the provider configuration is working"),
        )
        self.test_success_icon = get_icon("emblem-ok-symbolic")
        self.test_success_icon.add_css_class("success")
        self.test_success_icon.set_valign(Gtk.Align.CENTER)
        self.test_success_icon.set_visible(False)
        self.test_row.add_suffix(self.test_success_icon)

        self.test_btn = Gtk.Button(label=_("Test"))
        self.test_btn.add_css_class("pill")
        self.test_btn.set_valign(Gtk.Align.CENTER)
        self.test_btn.connect("clicked", self._on_test_clicked)
        self.test_row.add_suffix(self.test_btn)

        self.test_error_icon = get_icon("dialog-warning-symbolic")
        self.test_error_icon.add_css_class("error")
        self.test_error_icon.set_valign(Gtk.Align.CENTER)
        self.test_error_icon.set_visible(False)
        self.test_row.add_suffix(self.test_error_icon)

        self.add(self.test_row)

        self._clear_form()

    def set_provider(
        self, provider_id: Optional[str], config: Optional[AIProviderConfig]
    ):
        self._updating = True
        self.provider_id = provider_id

        if config:
            self.set_title(_("Edit Provider"))
            self.name_row.set_text(config.name)
            self.api_key_row.set_text(config.api_key)
            self.base_url_row.set_text(config.base_url)
            self.model_row.set_text(config.default_model)

            if config.provider_type == AIProviderType.OPENAI_COMPATIBLE:
                self.type_row.set_selected(0)
        else:
            self._clear_form()
            self.set_title(_("Provider Settings"))

        self._clear_test_status()
        self.set_visible(True)
        self._updating = False

    def _clear_form(self):
        self.name_row.set_text("")
        self.api_key_row.set_text("")
        self.base_url_row.set_text("")
        self.model_row.set_text("")
        self.type_row.set_selected(0)
        self._clear_test_status()

    def _clear_test_status(self):
        self.test_row.set_subtitle(
            _("Verify the provider configuration is working")
        )
        self.test_success_icon.set_visible(False)
        self.test_error_icon.set_visible(False)
        self.test_btn.set_sensitive(True)
        return False

    def _get_provider_type(self) -> AIProviderType:
        idx = self.type_row.get_selected()
        types = [AIProviderType.OPENAI_COMPATIBLE]
        return types[idx]

    def _on_name_changed(self, entry_row):
        if self._updating or not self.provider_id:
            return

        name = entry_row.get_text().strip()
        if not name:
            return

        row = self.list_widget.get_row_for_provider(self.provider_id)
        if row:
            row.title_label.set_label(name)

        self._save_config()

    def _on_field_changed(self, *args):
        if self._updating or not self.provider_id:
            return

        name = self.name_row.get_text().strip()
        if not name:
            return

        self._save_config()

    def _save_config(self):
        if not self.provider_id:
            return

        ai_service = get_context().ai_service
        current_config = ai_service.get_config(self.provider_id)
        if not current_config:
            return

        name = self.name_row.get_text().strip()
        api_key = self.api_key_row.get_text().strip()
        base_url = self.base_url_row.get_text().strip()
        default_model = self.model_row.get_text().strip()

        new_config = AIProviderConfig(
            id=self.provider_id,
            name=name,
            provider_type=self._get_provider_type(),
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            enabled=current_config.enabled,
        )

        self._updating = True
        ai_service.update_provider(new_config)
        self._updating = False

        row = self.list_widget.get_row_for_provider(self.provider_id)
        if row:
            row.config = new_config

    def _on_test_clicked(self, button):
        self.test_row.set_subtitle(_("Testing..."))
        self.test_success_icon.set_visible(False)
        self.test_error_icon.set_visible(False)
        self.test_btn.set_sensitive(False)

        async def do_test():
            try:
                test_config = AIProviderConfig(
                    id="test",
                    name=self.name_row.get_text() or "Test",
                    provider_type=self._get_provider_type(),
                    api_key=self.api_key_row.get_text(),
                    base_url=self.base_url_row.get_text(),
                    default_model=self.model_row.get_text(),
                    enabled=True,
                )

                from ...core.ai.openai_provider import OpenAICompatibleProvider

                provider = OpenAICompatibleProvider(test_config)
                success, message = await provider.test_connection()
                await provider.close()

                return success, message
            except Exception as e:
                return False, str(e)

        from ...shared.tasker import task_mgr

        future = asyncio.run_coroutine_threadsafe(do_test(), task_mgr.loop)

        def on_test_done(f):
            try:
                success, message = f.result()
                GLib.idle_add(self._update_test_result, success, message)
            except Exception as e:
                GLib.idle_add(self._update_test_result, False, str(e))

        future.add_done_callback(on_test_done)

    def _update_test_result(self, success: bool, message: str):
        if success:
            self.test_row.set_subtitle("")
            self.test_success_icon.set_visible(True)
            self.test_error_icon.set_visible(False)
            GLib.timeout_add_seconds(5, self._clear_test_status)
        else:
            self.test_row.set_subtitle(message)
            self.test_success_icon.set_visible(False)
            self.test_error_icon.set_visible(True)
        self.test_btn.set_sensitive(True)


class AISettingsPage(TrackedPreferencesPage):
    """Settings page for configuring AI providers."""

    key = "ai"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("AI"))
        self.set_icon_name("ai-symbolic")

        self.provider_list = ProviderListWidget(
            title=_("AI Providers"),
            description=_(
                "Configure AI providers for use by addons. "
                "Addons can use these providers without needing "
                "their own API keys."
            ),
        )
        self.add(self.provider_list)

        self.provider_editor = ProviderEditorWidget(
            self.provider_list, visible=False
        )
        self.add(self.provider_editor)

        self.provider_list.provider_selected.connect(
            self._on_provider_selected
        )

        get_context().ai_service.changed.connect(self._on_service_changed)

        self.provider_list.populate_and_select()

    def _on_provider_selected(
        self,
        sender,
        provider_id: Optional[str],
        config: Optional[AIProviderConfig],
    ):
        if provider_id and config:
            self.provider_editor.set_provider(provider_id, config)
        else:
            self.provider_editor.set_visible(False)

    def _on_service_changed(self, sender):
        if self.provider_editor._updating:
            return
        GLib.idle_add(self._refresh_after_change)

    def _refresh_after_change(self):
        current_id = self.provider_editor.provider_id
        self.provider_list.populate_and_select(select_id=current_id)
