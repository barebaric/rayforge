import asyncio
import logging
import webbrowser
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal, Optional

import aiohttp
import dacite

from .driver import DeviceStatus


# from ..transport import TransportStatus

logger = logging.getLogger(__name__)


@dataclass
class OctoprintNodeType:
    url: str
    version: str
    port: Optional[int] = None


class SocketMessagesTypes(Enum):
    CONNECTED = "connected"
    REAUTH_REQUIRED = "reauthRequired"
    CURRENT = "current"
    HISTORY = "history"
    EVENT = "event"
    SLICING_PROGRESS = "slicingProgress"
    PLUGIN = "plugin"


class ConnectionFlags(Enum):
    CANCELLING = "cancelling"
    CLOSED_OR_ERROR = "closedOrError"
    ERROR = "error"
    FINISHING = "finishing"
    OPERATIONAL = "operational"
    PAUSED = "paused"
    PAUSING = "pausing"
    PRINTING = "printing"
    READY = "ready"
    RESUMING = "resuming"
    SD_READY = "sdReady"


FLAGS_PRIORITY: list[ConnectionFlags] = [
    ConnectionFlags.ERROR,
    ConnectionFlags.CLOSED_OR_ERROR,
    ConnectionFlags.CANCELLING,
    ConnectionFlags.FINISHING,
    ConnectionFlags.PAUSING,
    ConnectionFlags.PAUSED,
    ConnectionFlags.RESUMING,
    ConnectionFlags.PRINTING,
    ConnectionFlags.OPERATIONAL,
    ConnectionFlags.READY,
    ConnectionFlags.SD_READY,
]

FLAGS_TO_STATUS: dict[ConnectionFlags, DeviceStatus] = {
    ConnectionFlags.ERROR: DeviceStatus.ALARM,
    ConnectionFlags.CLOSED_OR_ERROR: DeviceStatus.ALARM,
    ConnectionFlags.CANCELLING: DeviceStatus.HOLD,
    ConnectionFlags.FINISHING: DeviceStatus.CYCLE,
    ConnectionFlags.PAUSING: DeviceStatus.HOLD,
    ConnectionFlags.PAUSED: DeviceStatus.HOLD,
    ConnectionFlags.RESUMING: DeviceStatus.CYCLE,
    ConnectionFlags.PRINTING: DeviceStatus.RUN,
    ConnectionFlags.OPERATIONAL: DeviceStatus.IDLE,
    ConnectionFlags.READY: DeviceStatus.IDLE,
    ConnectionFlags.SD_READY: DeviceStatus.IDLE,
}


def resolve_status(flags: list[ConnectionFlags]) -> DeviceStatus:
    """Return the DeviceStatus for the highest-priority active flag."""
    flag_set = set(flags)
    for flag in FLAGS_PRIORITY:
        if flag in flag_set:
            return FLAGS_TO_STATUS[flag]
    return DeviceStatus.UNKNOWN


# TODO: Convert to Pydantic models if possible


@dataclass
class ConnectionInfos:
    connected: bool
    printer_name: str
    flags: List[ConnectionFlags]


@dataclass
class FileInfos:
    name: str
    display: str
    path: str
    type: Literal["model"] | Literal["machinecode"] | Literal["folder"]
    type_path: str
    origin: Literal["local"] | Literal["sdcard"]
    user: str


@dataclass
class FilamentInfos:
    length: float
    volume: float


@dataclass
class JobInfos:
    file: FileInfos
    estimated_print_time: float
    last_print_time: float
    filament: FilamentInfos

    @staticmethod
    def from_dict(d: dict) -> "JobInfos":
        return dacite.from_dict(data_class=JobInfos, data=d)


class AuthorizationWorkflow:
    APP_NAME = "Rayforge"
    app_token: Optional[str] = None
    poll_url = None

    def __init__(self, base_url: str, usr_name: str) -> None:
        self.base_url = base_url
        self.usr_name = usr_name

    async def probe_for_workflow(self) -> bool:
        async with aiohttp.ClientSession(base_url=self.base_url) as client:
            async with client.get("/plugin/appkeys/probe") as response:
                return response.status == 204

    async def request_authorization(self) -> str:
        payload = {"app": self.APP_NAME, "user": self.usr_name}
        async with aiohttp.ClientSession(base_url=self.base_url) as client:
            async with client.post(
                "/plugin/appkeys/request", json=payload
            ) as response:
                if response.status != 201:
                    raise Exception(
                        f"Failed to obtain API key: {response.status}"
                    )
                data = await response.json()
                self.app_token = data.get("app_token")
                self.poll_url = response.headers.get("Location", "")
                return data.get("auth_dialog")

    async def get_api_key(
        self, abort_event: Optional[asyncio.Event] = None
    ) -> str:
        if not self.app_token or not self.poll_url:
            raise Exception(
                "App token is not set. Please request authorization first."
            )

        while True:
            logger.debug("Polling for API key...")

            if abort_event and abort_event.is_set():
                raise Exception("Authorization aborted")
            async with aiohttp.ClientSession(base_url=self.base_url) as client:
                async with client.get(self.poll_url) as response:
                    if response.status == 202:
                        # Still waiting for user confirmation
                        await asyncio.sleep(1)
                        continue

                    elif response.status == 200:
                        data = await response.json()
                        if data:
                            self.api_key = data.get("api_key")
                            return self.api_key

                    elif response.status == 404:
                        return "User denied the request"

                await asyncio.sleep(1)

    async def run_workflow(self) -> str:
        if await self.probe_for_workflow():
            auth_url = await self.request_authorization()
            if auth_url:
                webbrowser.open(auth_url)
                logger.info(f"Authorization URL: {auth_url}")
            else:
                logger.error("Failed to get authorization URL")
            api_key = await self.get_api_key()
            return api_key
        else:
            raise Exception(
                "Authorization workflow is not "
                "supported on this Octoprint instance"
            )
