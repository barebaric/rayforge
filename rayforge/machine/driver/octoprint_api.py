import asyncio
import logging
import aiohttp
import webbrowser

from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class OctoprintNodeType:
    url: str
    version: str
    port: Optional[int] = None


class AuthorizationWorkflow:
    APP_NAME = "Rayforge"
    app_token: Optional[str] = None
    poll_url = None

    def __init__(self, base_url: str, usr_name: str) -> None:
        self.base_url = base_url
        self.usr_name = usr_name
        self.http_client: Optional[aiohttp.ClientSession] = None

    async def _get_http_client(self) -> aiohttp.ClientSession:
        """Lazily create and return the HTTP client."""
        if self.http_client is None:
            self.http_client = aiohttp.ClientSession(base_url=self.base_url)
        return self.http_client

    async def probe_for_workflow(self) -> bool:
        client = await self._get_http_client()
        async with client.get("/plugin/appkeys/probe") as response:
            return response.status == 204

    async def request_authorization(self) -> str:
        payload = {"app": self.APP_NAME, "user": self.usr_name}
        client = await self._get_http_client()
        async with client.post(
            "/plugin/appkeys/request", json=payload
        ) as response:
            if response.status != 201:
                raise Exception(f"Failed to obtain API key: {response.status}")
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

            client = await self._get_http_client()
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
            print(api_key)
            return api_key
        else:
            raise Exception(
                "Authorization workflow is not "
                "supported on this Octoprint instance"
            )
