"""Shared remote-endpoint setup vars and session-config mapping.

Remote is a configuration choice on the migrated driver shells:
the same driver class works against locally attached hardware or
against a raydriver daemon next to the machine. The session
factory in ``raydriver.remote.client`` routes on the keys added
here; everything else in the shells stays unchanged.
"""

from gettext import gettext as _
from typing import Any

from ...core.varset import Var
from ...core.varset.hostnamevar import hostname_validator

#: Setup-var keys that describe the remote endpoint.
REMOTE_KEYS = (
    "remote",
    "remote_host",
    "remote_port",
    "token",
    "tls_ca",
    "tls_hostname",
)


def is_remote(kwargs: dict[str, Any]) -> bool:
    """True when the setup selects a remote daemon endpoint."""
    return bool(kwargs.get("remote"))


def _when_remote(values: dict[str, Any]) -> bool:
    return bool(values.get("remote"))


def remote_setup_vars() -> list[Var[Any]]:
    """The optional remote-endpoint vars for a driver's VarSet."""
    return [
        Var(
            key="remote",
            label=_("Remote device"),
            description=_(
                "Drive a machine attached to a raydriver daemon on "
                "another computer instead of locally attached "
                "hardware."
            ),
            var_type=bool,
            default=False,
        ),
        Var(
            key="remote_host",
            label=_("Daemon hostname"),
            description=_(
                "Hostname or IP of the computer running the raydriver daemon."
            ),
            var_type=str,
            validator=hostname_validator,
            visible_when=_when_remote,
            required_when=_when_remote,
        ),
        Var(
            key="remote_port",
            label=_("Daemon port"),
            description=_("Control port of the raydriver daemon."),
            var_type=int,
            default=8765,
            visible_when=_when_remote,
            required_when=_when_remote,
        ),
        Var(
            key="token",
            label=_("Token"),
            description=_("Shared secret configured on the daemon."),
            var_type=str,
            default="",
            optional=True,
            visible_when=_when_remote,
        ),
        Var(
            key="tls_ca",
            label=_("TLS certificate (PEM path)"),
            description=_(
                "Pin the daemon's certificate. Required when a token is used."
            ),
            var_type=str,
            default="",
            optional=True,
            visible_when=_when_remote,
        ),
        Var(
            key="tls_hostname",
            label=_("TLS hostname"),
            description=_(
                "Hostname to verify against the certificate "
                "(defaults to the daemon hostname)."
            ),
            var_type=str,
            default="",
            optional=True,
            visible_when=_when_remote,
        ),
    ]


def remote_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remote session-config keys extracted from the setup vars."""
    if not is_remote(kwargs):
        return {}
    config: dict[str, Any] = {
        "remote_host": kwargs.get("remote_host", ""),
        "remote_port": int(kwargs.get("remote_port", 8765) or 8765),
    }
    for key in ("token", "tls_ca", "tls_hostname"):
        value = kwargs.get(key) or ""
        if value:
            config[key] = value
    return config
