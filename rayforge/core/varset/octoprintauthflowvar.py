from __future__ import annotations
import asyncio
import threading
from typing import Callable, Optional

from gettext import gettext as _

from rayforge.core.varset.hostnamevar import HostnameVar
from rayforge.core.varset.portvar import PortVar
from .var import Var


class OctoprintAuthFlowVar(Var[str]):
    """A variable that represents a button for the octoprint auth flow."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        """
        Initializes a new OctoprintAuthFlowVar instance.
        Value format: token@host:port

        Args:
            key: The unique machine-readable identifier.
            label: The human-readable name for the UI.
            description: A longer, human-readable description.
            default: The default value (The octoprint token).
            value: The initial value. If provided, it overrides the default.
        """
        self._token = ""
        self.host_var = HostnameVar(
            key="host",
            label=_("Hostname"),
            description=_("The IP address or hostname of the device"),
            default="",
        )
        self.port_var = PortVar(
            key="port",
            label=_("HTTP Port"),
            description=_("The HTTP port for the device"),
            default=80,
        )
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
        )

    @staticmethod
    def parse_value(val: str) -> tuple[str, str, int]:
        token_part, host_port_part = val.split("@")
        host_part, port_part = host_port_part.split(":")
        return token_part, host_part, int(port_part)

    @property
    def value(self) -> Optional[str]:
        """The value is a string in the format token@host:port"""
        return f"{self._token}@{self.host_var.value}:{self.port_var.value}"

    @property
    def raw_value(self) -> Optional[str]:
        """Returns just the token part of the value."""
        return self._token

    @value.setter
    def value(self, new_value: Optional[str]):
        """Parse the value and set the token, host, and port accordingly."""
        if new_value is None:
            self._token = ""
            self.host_var.value = ""
            self.port_var.value = 80
            return

        try:
            token_part, host_port_part = new_value.split("@")
            host_part, port_part = host_port_part.split(":")
            self._token = token_part
            self.host_var.value = host_part
            self.port_var.value = int(port_part)
        except ValueError:
            raise ValueError(
                "Invalid value format. Expected 'token@host:port'."
            )

    def _on_click(self, finish_callback: Optional[Callable] = None):
        """Starts the authorization workflow in a background thread."""
        from rayforge.machine.driver.octoprint_api import AuthorizationWorkflow

        host = self.host_var.value
        port = self.port_var.value

        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                auth_workflow = AuthorizationWorkflow(
                    base_url=f"http://{host}:{port}/", usr_name=""
                )
                token = loop.run_until_complete(auth_workflow.run_workflow())
                self._token = token
                if finish_callback:
                    finish_callback()
            finally:
                loop.close()

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
