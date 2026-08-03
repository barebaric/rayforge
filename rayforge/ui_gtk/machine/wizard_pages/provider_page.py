"""Step 5 — AI provider configuration (shown only when needed).

Inserted between the probe step and the AI spec lookup step when no AI
provider is configured yet. Lets the user point the wizard at an
OpenAI-compatible endpoint (base URL + API key) so Step 6 can query it
for known machine specifications. Skipping the page routes straight to
manual entry (Step 7), skipping the AI lookup entirely.
"""

import uuid
from gettext import gettext as _

from gi.repository import Adw

from ....context import get_context
from ....core.ai.provider import AIProviderConfig, AIProviderType
from ....machine.device.profile import DeviceProfile
from . import WizardPage, _makePreferencesGroup

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


class AIProviderPage(WizardPage):
    step_number = 5
    title = _("AI Provider")
    subtitle = _(
        "Configure an AI provider so the wizard can pre-fill "
        "known machine specifications."
    )

    def __init__(self, wizard, **kwargs):
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        group = _makePreferencesGroup(
            title=_("AI Provider"),
            description=_(
                "Enter an OpenAI-compatible endpoint. This is only "
                "used for the automatic spec lookup; you can also "
                "skip and enter the values by hand."
            ),
        )
        self.content.append(group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.set_text(_("Default Provider"))
        self.name_row.connect("changed", self._on_inputs_changed)
        group.add(self.name_row)

        self.base_url_row = Adw.EntryRow(title=_("Base URL"))
        self.base_url_row.set_text(_DEFAULT_BASE_URL)
        self.base_url_row.connect("changed", self._on_inputs_changed)
        group.add(self.base_url_row)

        self.api_key_row = Adw.PasswordEntryRow(title=_("API Key"))
        self.api_key_row.connect("changed", self._on_inputs_changed)
        group.add(self.api_key_row)

        self.model_row = Adw.EntryRow(title=_("Default Model (optional)"))
        self.model_row.connect("changed", self._on_inputs_changed)
        group.add(self.model_row)

        # Ready only when the essential fields are filled in, so Next
        # ("use this provider") and Skip ("no AI, enter values by hand")
        # stay semantically distinct.
        self._refresh_ready()

    def _on_inputs_changed(self, _row, _param=None) -> None:
        self._refresh_ready()

    def _refresh_ready(self) -> None:
        ready = bool(
            self.name_row.get_text().strip()
            and self.base_url_row.get_text().strip()
            and self.api_key_row.get_text().strip()
        )
        self.set_ready(ready)

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        name = self.name_row.get_text().strip()
        base_url = self.base_url_row.get_text().strip()
        api_key = self.api_key_row.get_text().strip()
        default_model = self.model_row.get_text().strip()
        if not (name and base_url and api_key):
            return True

        config = AIProviderConfig(
            id=str(uuid.uuid4())[:8],
            name=name,
            provider_type=AIProviderType.OPENAI_COMPATIBLE,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            enabled=True,
        )
        get_context().ai_service.add_provider(config)
        return True


__all__ = ["AIProviderPage"]
