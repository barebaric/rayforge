"""
Dialog for importing a machine from a remote rayforge-server.

Creates a local mirror of a machine hosted by a
:mod:`rayforge.machine.remote.server` instance: the profile
(kinematics, heads, dialect) is copied, and the mirror is driven by
:class:`~rayforge.machine.remote.driver.RemoteDriver`, so jobs are
computed locally and executed by the server.
"""

import logging
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from gi.repository import Adw, Gtk

from ...context import get_context
from ...machine.remote import DEFAULT_PORT
from ...machine.remote.sync import (
    fetch_machine_config,
    fetch_server_machines,
    import_remote_machine,
)
from ...shared.tasker import task_mgr
from ...shared.tasker.context import ExecutionContext

if TYPE_CHECKING:
    from ...machine.models.machine import Machine

logger = logging.getLogger(__name__)


class RemoteImportDialog(Adw.MessageDialog):
    """Imports a machine profile hosted by a rayforge-server."""

    def __init__(
        self,
        on_imported: Callable[["Machine"], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            modal=True,
            heading=_("Import from rayforge-server"),
            body=_(
                "Enter the address of a machine's rayforge-server. "
                "The machine is copied to this computer and its jobs "
                "are executed by the server, so this computer can be "
                "closed while a job is running."
            ),
            **kwargs,
        )
        self._on_imported = on_imported
        self._machines: list[dict[str, Any]] = []
        self._host = ""
        self._port = DEFAULT_PORT

        self.set_size_request(460, -1)
        self._build_extra_child()

        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")
        self.set_close_response("cancel")
        self.add_response("fetch", _("Connect"))
        self.set_response_appearance("fetch", Adw.ResponseAppearance.SUGGESTED)
        self.add_response("import", _("Import Machine"))
        self.set_response_appearance(
            "import", Adw.ResponseAppearance.SUGGESTED
        )
        self.set_response_enabled("import", False)
        self.connect("response", self._on_response)

    # -- UI construction --------------------------------------------------

    def _build_extra_child(self) -> None:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_top(12)

        self.host_row = Adw.EntryRow(title=_("Hostname or IP address"))
        self.port_row = Adw.EntryRow(title=_("Port"))
        self.port_row.set_text(str(DEFAULT_PORT))
        self.port_row.set_input_purpose(Gtk.InputPurpose.NUMBER)

        entries = Adw.PreferencesGroup()
        entries.add(self.host_row)
        entries.add(self.port_row)
        box.append(entries)

        self.error_label = Gtk.Label(xalign=0.0, wrap=True)
        self.error_label.add_css_class("error")
        self.error_label.set_visible(False)
        box.append(self.error_label)

        self.machine_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6
        )
        self.machine_box.set_visible(False)
        machine_label = Gtk.Label(
            label=_("Machines on the server:"), xalign=0.0
        )
        machine_label.add_css_class("caption-heading")
        self.machine_box.append(machine_label)
        self.machine_dropdown = Gtk.DropDown()
        self.machine_box.append(self.machine_dropdown)
        box.append(self.machine_box)

        self.set_extra_child(box)

    def _show_error(self, message: str) -> None:
        """Displays *message* inside the dialog."""
        self.error_label.set_text(message)
        self.error_label.set_visible(True)

    # -- Event handling ---------------------------------------------------

    def _on_response(self, _dialog, response: str) -> None:
        if response == "fetch":
            self._fetch_machines()
        elif response == "import":
            self._import_machine()

    def _fetch_machines(self) -> None:
        """Fetches the machine list from the server in the background."""
        host = self.host_row.get_text().strip()
        try:
            port = int(self.port_row.get_text().strip())
        except ValueError:
            self._show_error(_("The port must be a number."))
            return
        if not host:
            self._show_error(_("Enter the hostname or IP address."))
            return

        self._host = host
        self._port = port
        self.error_label.set_visible(False)
        self.set_response_enabled("fetch", False)

        async def _coroutine(exec_ctx: ExecutionContext) -> Any:
            return await fetch_server_machines(host, port)

        task_mgr.add_coroutine(
            _coroutine,
            key=f"remote-import-fetch-{id(self)}",
            when_done=lambda task: self._on_machines_fetched(task),
        )

    def _on_machines_fetched(self, task: Any) -> None:
        """Marshals the fetched machine list into the dialog."""
        task_mgr.schedule_on_main_thread(lambda: self._apply_machines(task))

    def _apply_machines(self, task: Any) -> None:
        try:
            machines = task.result()
        except Exception as e:
            logger.debug("Fetching remote machines failed", exc_info=True)
            self.set_response_enabled("fetch", True)
            self._show_error(str(e))
            return

        if not machines:
            self.set_response_enabled("fetch", True)
            self._show_error(_("The server hosts no machines."))
            return

        self._machines = machines
        labels = [
            f"{m.get('name', 'Machine')} ({m.get('id', '')})" for m in machines
        ]
        self.machine_dropdown.set_model(
            Gtk.StringList.new(labels),
        )
        self.machine_dropdown.set_selected(0)
        self.machine_box.set_visible(True)
        self.set_response_enabled("import", True)
        if len(machines) == 1:
            # Nothing to choose: behave like a confirm dialog.
            self.set_response_enabled("fetch", False)

    def _selected_machine_id(self) -> str:
        index = self.machine_dropdown.get_selected()
        if 0 <= index < len(self._machines):
            return str(self._machines[index].get("id", ""))
        return ""

    def _import_machine(self) -> None:
        """Fetches the machine config and creates the local mirror."""
        machine_id = self._selected_machine_id()
        self.set_response_enabled("import", False)

        async def _coroutine(exec_ctx: ExecutionContext) -> Any:
            config = await fetch_machine_config(
                self._host, self._port, machine_id
            )
            return import_remote_machine(
                get_context().machine_mgr,
                config,
                self._host,
                self._port,
                machine_id,
            )

        task_mgr.add_coroutine(
            _coroutine,
            key=f"remote-import-{id(self)}",
            when_done=lambda task: self._on_import_done(task),
        )

    def _on_import_done(self, task: Any) -> None:
        task_mgr.schedule_on_main_thread(lambda: self._apply_import(task))

    def _apply_import(self, task: Any) -> None:
        try:
            machine = task.result()
        except Exception as e:
            logger.exception("Importing remote machine failed")
            self.set_response_enabled("import", True)
            self._show_error(str(e))
            return
        if self._on_imported is not None:
            self._on_imported(machine)
        self.close()
