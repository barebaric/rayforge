import json
from collections.abc import Callable
from typing import Any

from .var import Var


class AppKeyVar(Var[str]):
    """
    A Var for obtaining an API key from a device that supports a
    decision-based key approval flow.

    The value is a JSON string containing the key data, or "" if
    not yet obtained.  Users can also enter a key manually.

    URL config fields support ``{key}`` templates that reference
    sibling Vars in the same VarSet, resolved at request time.
    """

    display_name = "Application Key"

    def __init__(
        self,
        key: str,
        label: str,
        app_name: str,
        probe_url: str | None = None,
        request_url: str | None = None,
        poll_url: str | None = None,
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        self.app_name = app_name
        self.probe_url = probe_url
        self.request_url = request_url
        self.poll_url = poll_url
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default or "",
            value=value,
            visible_when=visible_when,
        )

    def get_api_key(self) -> str | None:
        val = self.value
        if not val:
            return None
        try:
            tokens = json.loads(val)
            if isinstance(tokens, dict):
                return tokens.get("api_key")
            return str(tokens)
        except (json.JSONDecodeError, TypeError):
            return val.strip() if val else None

    def has_key(self) -> bool:
        return bool(self.get_api_key())

    def resolve_config(
        self,
        overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        sibling_values: dict[str, Any] = {}
        if self._varset is not None:
            sibling_values = self._varset.get_values()

        merged: dict[str, Any] = {
            "app_name": self.app_name,
            "probe_url": self.probe_url,
            "request_url": self.request_url,
            "poll_url": self.poll_url,
        }
        if overrides:
            for k, v in overrides.items():
                if v and v.strip():
                    merged[k] = v

        def _resolve(template):
            if template is None:
                return None
            return template.format(**sibling_values)

        merged["probe_url"] = _resolve(merged["probe_url"])
        merged["request_url"] = _resolve(merged["request_url"])
        merged["poll_url"] = _resolve(merged["poll_url"])
        return merged

    def to_dict(self, include_value: bool = False) -> dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data["app_name"] = self.app_name
        data["probe_url"] = self.probe_url
        data["request_url"] = self.request_url
        data["poll_url"] = self.poll_url
        return data
