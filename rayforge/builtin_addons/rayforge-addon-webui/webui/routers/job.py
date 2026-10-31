"""Job endpoints: start, hold, cancel, status."""

from gettext import gettext as _

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from rayforge.machine.cmd import JobAlreadyRunningError
from rayforge.machine.job_status import JobStatus

from .deps import conflict, get_bridge, machine_unavailable

router = APIRouter(tags=["job"])


class HoldRequest(BaseModel):
    hold: bool


def _serialize_job_status(status: JobStatus) -> dict:
    return {
        "machine_id": status.machine_id,
        "machine_name": status.machine_name,
        "kind": status.kind.name.lower(),
        "phase": status.phase.name.lower(),
        "started_at": status.started_at,
        "elapsed_seconds": status.elapsed_seconds,
        "total_distance_mm": status.total_distance,
        "traveled_distance_mm": status.traveled_distance,
        "progress_fraction": status.progress_fraction,
        "eta_seconds": status.eta_seconds,
        "estimated_seconds": status.estimated_seconds,
    }


@router.post("/job/start")
async def start_job(request: Request) -> dict:
    bridge = get_bridge(request)

    def _start(editor):
        machine = editor.context.machine
        if machine is None:
            raise machine_unavailable()
        machine_cmd = request.app.state.machine_cmd
        events = request.app.state.events
        return machine_cmd.send_job(
            machine, on_progress=events.on_job_progress
        )

    try:
        await bridge.run(_start, bridge.editor)
    except JobAlreadyRunningError as e:
        raise conflict(str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=_("Job could not be started: {error}").format(error=e),
        ) from e
    return {"status": "accepted"}


@router.post("/job/hold")
async def hold_job(request: Request, payload: HoldRequest) -> dict:
    bridge = get_bridge(request)

    def _hold(editor):
        machine = editor.context.machine
        if machine is None:
            raise machine_unavailable()
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.set_hold(machine, payload.hold)

    await bridge.run(_hold, bridge.editor)
    return {"status": "accepted", "hold": payload.hold}


@router.post("/job/cancel")
async def cancel_job(request: Request) -> dict:
    bridge = get_bridge(request)

    def _cancel(editor):
        machine = editor.context.machine
        if machine is None:
            raise machine_unavailable()
        machine_cmd = request.app.state.machine_cmd
        machine_cmd.cancel_job(machine)

    await bridge.run(_cancel, bridge.editor)
    return {"status": "accepted"}


@router.get("/job/status")
async def get_job_status(request: Request) -> dict:
    bridge = get_bridge(request)

    def _status(editor):
        machine = editor.context.machine
        if machine is None:
            raise machine_unavailable()
        current = machine.current_job
        return {
            "active": current is not None,
            "job": (
                _serialize_job_status(current) if current is not None else None
            ),
        }

    return await bridge.run(_status, bridge.editor)
