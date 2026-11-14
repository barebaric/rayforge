"""Daemon-endpoint setup vars for the generic remote driver.

A remote machine is always remote: there is no local/remote toggle
and no per-driver remote wiring. The endpoint vars here describe
the daemon to connect to; the device itself is chosen with a device
picker fed by the daemon. The protocol is the daemon's knowledge,
resolved by raydriver and surfaced as info — never selected here.
"""

from gettext import gettext as _
from typing import Any

from raydriver.analyzer.listener import is_loopback

from ...core.varset import Var
from ...core.varset.hostnamevar import hostname_validator

#: Endpoint keys consumed by raydriver.remote.client.
ENDPOINT_KEYS = (
    "remote_host",
    "remote_port",
    "token",
    "tls_ca",
    "tls_hostname",
)


def _when_tls_needed(values: dict[str, Any]) -> bool:
    """TLS settings only apply to a daemon reached over the network.

    A loopback daemon needs no certificate (the token never leaves
    the machine), so hide the TLS fields for it.
    """
    host = values.get("remote_host") or ""
    return bool(host) and not is_loopback(host)


def endpoint_vars() -> list[Var[Any]]:
    """The daemon-endpoint vars for the generic remote driver."""
    return [
        Var(
            key="remote_host",
            label=_("Daemon hostname"),
            description=_(
                "Hostname or IP of the computer running the raydriver daemon."
            ),
            var_type=str,
            validator=hostname_validator,
        ),
        Var(
            key="remote_port",
            label=_("Daemon port"),
            description=_("Control port of the raydriver daemon."),
            var_type=int,
            default=8765,
        ),
        Var(
            key="token",
            label=_("Token"),
            description=_(
                "The daemon's shared secret. Run 'raydriver daemon status' "
                "on the computer running the daemon to print it."
            ),
            var_type=str,
            default="",
            optional=True,
        ),
        Var(
            key="tls_ca",
            label=_("TLS certificate (PEM path)"),
            description=_(
                "Pin the daemon's certificate. Only needed for a daemon "
                "reached over the network with TLS."
            ),
            var_type=str,
            default="",
            optional=True,
            visible_when=_when_tls_needed,
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
            visible_when=_when_tls_needed,
        ),
    ]


def endpoint_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The endpoint dict consumed by raydriver.remote.client."""
    config: dict[str, Any] = {
        "remote_host": kwargs.get("remote_host", ""),
        "remote_port": int(kwargs.get("remote_port", 8765) or 8765),
    }
    for key in ("token", "tls_ca", "tls_hostname"):
        value = kwargs.get(key) or ""
        if value:
            config[key] = value
    return config
