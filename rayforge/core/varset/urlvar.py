from typing import Optional, Tuple
from gettext import gettext as _
from urllib.parse import urlparse
from .var import Var, ValidationError


def url_validator(
    url: Optional[str], allowed_schemes: Optional[Tuple[str, ...]] = None
):
    """
    Raises ValidationError if the string is not a valid URL.

    Args:
        url: The URL string to validate.
        allowed_schemes: Optional tuple of allowed schemes
            (e.g., ('http', 'https')). If None, any scheme is allowed.
    """
    if not url:
        raise ValidationError(_("URL cannot be empty."))
    try:
        parsed = urlparse(url)
        if not parsed.scheme:
            raise ValidationError(
                _("URL must include a scheme (e.g., 'http://').")
            )
        if not parsed.netloc:
            raise ValidationError(_("URL must include a hostname."))
        if allowed_schemes and parsed.scheme not in allowed_schemes:
            schemes_str = ", ".join(f"'{s}://'" for s in allowed_schemes)
            raise ValidationError(
                _("URL scheme must be one of: {schemes}.").format(
                    schemes=schemes_str
                )
            )
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(_("Invalid URL: {error}").format(error=str(e)))


class UrlVar(Var[str]):
    """A Var subclass for generic URLs."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
        allowed_schemes: Optional[Tuple[str, ...]] = None,
    ):
        self.allowed_schemes = allowed_schemes

        def validator(url: Optional[str]):
            url_validator(url, allowed_schemes=allowed_schemes)

        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
            validator=validator,
        )


class WebsocketUrlVar(Var[str]):
    """A Var subclass specifically for WebSocket URLs (ws:// or wss://)."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        def validator(url: Optional[str]):
            url_validator(url, allowed_schemes=("ws", "wss"))

        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
            validator=validator,
        )
