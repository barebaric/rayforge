"""UI tests for the remote machine import dialog."""

from typing import Any

import pytest

from rayforge.machine.remote import RemoteJobServer
from rayforge.machine.remote.sync import (
    fetch_machine_config,
    fetch_server_machines,
    import_remote_machine,
)
from rayforge.ui_gtk.machine.remote_import_dialog import RemoteImportDialog


def _task(result: Any = None, error: Exception | None = None) -> Any:
    """Mimics the TaskManager task interface the dialog consumes."""

    def _result():
        if error is not None:
            raise error
        return result

    return type("FakeTask", (), {"result": staticmethod(_result)})()


@pytest.mark.ui
@pytest.mark.asyncio
async def test_dialog_lists_fetched_machines(lite_context, task_mgr):
    server = RemoteJobServer(lite_context, task_mgr)
    await server.start()
    try:
        machines = await fetch_server_machines("127.0.0.1", server.port)
        assert len(machines) >= 1

        dialog = RemoteImportDialog()
        dialog._apply_machines(_task(result=machines))

        assert dialog.machine_box.get_visible()
        model = dialog.machine_dropdown.get_model()
        assert model is not None
        assert model.get_n_items() == len(machines)
        assert dialog._selected_machine_id() == machines[0]["id"]
    finally:
        await server.stop()


@pytest.mark.ui
def test_dialog_shows_fetch_error():
    dialog = RemoteImportDialog()

    dialog._apply_machines(_task(error=OSError("connection refused")))

    assert dialog.error_label.get_visible()
    assert "connection refused" in dialog.error_label.get_text()
    assert not dialog._machines


@pytest.mark.ui
def test_dialog_shows_empty_server_error():
    dialog = RemoteImportDialog()

    dialog._apply_machines(_task(result=[]))

    assert dialog.error_label.get_visible()
    assert not dialog.machine_box.get_visible()
    assert not dialog._machines


@pytest.mark.ui
def test_dialog_rejects_bad_port():
    dialog = RemoteImportDialog()
    dialog.port_row.set_text("not a port")

    dialog._fetch_machines()

    assert dialog.error_label.get_visible()
    assert not dialog._machines


@pytest.mark.ui
@pytest.mark.asyncio
async def test_dialog_imports_selected_machine(lite_context, task_mgr):
    server = RemoteJobServer(lite_context, task_mgr)
    await server.start()
    mirror = None
    imported = []
    try:
        machines = await fetch_server_machines("127.0.0.1", server.port)
        dialog = RemoteImportDialog(on_imported=imported.append)
        dialog._apply_machines(_task(result=machines))
        dialog._host = "127.0.0.1"
        dialog._port = server.port

        config = await fetch_machine_config(
            "127.0.0.1", server.port, dialog._selected_machine_id()
        )
        mirror = import_remote_machine(
            lite_context.machine_mgr,
            config,
            "127.0.0.1",
            server.port,
            dialog._selected_machine_id(),
        )

        dialog._apply_import(_task(result=mirror))

        assert len(imported) == 1
        assert imported[0] is mirror
    finally:
        if mirror is not None:
            await mirror.shutdown()
        await server.stop()
