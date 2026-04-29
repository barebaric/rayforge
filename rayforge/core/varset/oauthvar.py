import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from .var import Var


class OAuthFlowVar(Var[str]):
    """
    A Var that represents an OAuth 2.0 Authorization Code flow.

    The value is a JSON string containing token data, or "" if not
    yet authenticated.

    URL config fields support ``{key}`` templates that reference
    sibling Vars in the same VarSet, resolved at flow-start time.
    If a field is ``None`` the adapter shows an entry row so the
    user can provide it manually.
    """

    display_name = "OAuth Authentication"

    def __init__(
        self,
        key: str,
        label: str,
        authorize_url: Optional[str] = None,
        token_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        redirect_port: int = 8765,
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        self.authorize_url = authorize_url
        self.token_url = token_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.scopes = scopes or []
        self.redirect_port = redirect_port
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default or "",
            value=value,
        )

    def get_tokens(self) -> Optional[Dict[str, Any]]:
        """Return parsed token dict, or None if not authenticated."""
        val = self.value
        if not val:
            return None
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return None

    def is_authenticated(self) -> bool:
        tokens = self.get_tokens()
        if not tokens:
            return False
        if not tokens.get("access_token"):
            return False
        if self._is_expired(tokens):
            return False
        return True

    def is_expired(self) -> bool:
        tokens = self.get_tokens()
        if not tokens:
            return False
        return self._is_expired(tokens)

    def get_refresh_token(self) -> Optional[str]:
        tokens = self.get_tokens()
        if tokens:
            return tokens.get("refresh_token")
        return None

    @staticmethod
    def _is_expired(tokens: Dict[str, Any]) -> bool:
        expires_at_str = tokens.get("expires_at")
        if not expires_at_str:
            return False
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
            return datetime.now() >= expires_at
        except (ValueError, TypeError):
            return False

    def resolve_config(
        self,
        overrides: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Build a resolved config dict with all ``{key}`` placeholders
        substituted from sibling Var values.

        *overrides* is an optional dict (typically from user-provided
        entry rows in the adapter) that takes precedence over the
        var's own fields.
        """
        sibling_values: Dict[str, Any] = {}
        if self._varset is not None:
            sibling_values = self._varset.get_values()

        merged: Dict[str, Any] = {
            "authorize_url": self.authorize_url,
            "token_url": self.token_url,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "scopes": self.scopes,
            "redirect_port": self.redirect_port,
        }
        if overrides:
            for k, v in overrides.items():
                if v and v.strip():
                    merged[k] = v

        def _resolve(template):
            if template is None:
                return None
            try:
                return template.format(**sibling_values)
            except (KeyError, IndexError):
                return template

        merged["authorize_url"] = _resolve(merged["authorize_url"])
        merged["token_url"] = _resolve(merged["token_url"])
        merged["client_id"] = _resolve(merged["client_id"])
        merged["client_secret"] = _resolve(merged["client_secret"])
        return merged

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data["authorize_url"] = self.authorize_url
        data["token_url"] = self.token_url
        data["client_id"] = self.client_id
        data["client_secret"] = self.client_secret
        data["scopes"] = self.scopes
        data["redirect_port"] = self.redirect_port
        return data
