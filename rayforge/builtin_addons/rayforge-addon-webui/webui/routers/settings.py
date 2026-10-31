"""Application settings endpoints backed by the core ConfigManager."""

from gettext import gettext as _

from fastapi import APIRouter, Request
from pydantic import BaseModel

from rayforge.core.config import OpsColorMode

from .deps import bad_request, get_bridge

router = APIRouter(tags=["settings"])

OPS_COLOR_MODES = [mode.value for mode in OpsColorMode]

BOOLEAN_SETTINGS = (
    "watch_source_files",
    "check_for_app_updates",
)

FLOAT_SETTINGS = ("import_dpi",)

INT_SETTINGS = ("cache_budget_bytes",)

STRING_SETTINGS = (
    "theme",
    "ops_color_mode",
    "language",
)

UNIT_KEYS = ("length", "speed", "acceleration")


class SettingsPatch(BaseModel):
    values: dict


def _settings_payload(config) -> dict:
    return {
        "theme": config.theme,
        "language": config.language,
        "import_dpi": float(config.import_dpi),
        "cache_budget_bytes": int(config.cache_budget_bytes),
        "watch_source_files": config.watch_source_files,
        "check_for_app_updates": config.check_for_app_updates,
        "ops_color_mode": config.ops_color_mode.value,
        "unit_preferences": dict(config.unit_preferences),
    }


def _get_settings(editor) -> dict:
    return _settings_payload(editor.context.config)


@router.get("/settings")
async def get_settings(request: Request) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_get_settings, bridge.editor)


def _validate(key: str, value):
    if key in BOOLEAN_SETTINGS:
        if not isinstance(value, bool):
            raise bad_request(
                _("Setting {key} must be a boolean").format(key=key)
            )
        return value
    if key in FLOAT_SETTINGS:
        number = float(value)
        if not 1.0 <= number <= 4800.0:
            raise bad_request(
                _("Setting {key} is out of range").format(key=key)
            )
        return number
    if key in INT_SETTINGS:
        number = int(value)
        if number <= 0:
            raise bad_request(
                _("Setting {key} must be positive").format(key=key)
            )
        return number
    if key == "theme":
        if value not in ("system", "light", "dark"):
            raise bad_request(_("Unknown theme: {value}").format(value=value))
        return value
    if key == "ops_color_mode":
        if value not in OPS_COLOR_MODES:
            raise bad_request(
                _("Unknown color mode: {value}").format(value=value)
            )
        return value
    if key == "language":
        if value is not None and not isinstance(value, str):
            raise bad_request(_("Setting language must be a string or null"))
        return value
    if key == "unit_preferences":
        if not isinstance(value, dict):
            raise bad_request(_("Setting unit_preferences must be an object"))
        unknown = set(value) - set(UNIT_KEYS)
        if unknown:
            raise bad_request(
                _("Unknown unit preference: {key}").format(key=min(unknown))
            )
        return {str(k): str(v) for k, v in value.items()}
    raise bad_request(_("Unknown setting: {key}").format(key=key))


def _apply_settings(editor, values: dict) -> dict:
    config = editor.context.config
    for key, value in values.items():
        checked = _validate(key, value)
        if key == "theme":
            config.set_theme(checked)
        elif key == "language":
            config.set_language(checked)
        elif key == "import_dpi":
            config.set_import_dpi(checked)
        elif key == "cache_budget_bytes":
            config.set_cache_budget_bytes(checked)
        elif key == "watch_source_files":
            config.set_watch_source_files(checked)
        elif key == "check_for_app_updates":
            config.set_check_for_app_updates(checked)
        elif key == "ops_color_mode":
            config.ops_color_mode = OpsColorMode(checked)
            config.changed.send(config)
        elif key == "unit_preferences":
            config.unit_preferences.update(checked)
            config.changed.send(config)
    return _settings_payload(config)


@router.put("/settings")
async def put_settings(request: Request, patch: SettingsPatch) -> dict:
    bridge = get_bridge(request)
    return await bridge.run(_apply_settings, bridge.editor, patch.values)
