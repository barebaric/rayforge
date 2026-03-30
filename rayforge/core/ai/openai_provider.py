import json
import logging
from gettext import gettext as _
from typing import AsyncGenerator, List, Optional, Tuple

import aiohttp

from .provider import (
    AIProvider,
    AIProviderConfig,
    AIServiceError,
    ChatMessage,
    ChatResponse,
    make_api_error_message,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(AIProvider):
    """
    Provider for OpenAI and compatible APIs.

    Supports OpenAI, Ollama, LocalAI, and any OpenAI-compatible endpoint.
    """

    def __init__(self, config: AIProviderConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
            self._session = aiohttp.ClientSession(
                base_url=self.config.base_url, headers=headers
            )
        return self._session

    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        session = await self._get_session()
        payload = {
            "model": model or self.config.default_model,
            "messages": [m.to_dict() for m in messages],
            **kwargs,
        }

        try:
            async with session.post("chat/completions", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise AIServiceError(
                        make_api_error_message(resp.status, text)
                    )
                data = await resp.json()
        except aiohttp.ClientError as e:
            raise AIServiceError(
                _("Connection failed - please check your network")
            ) from e

        choice = data["choices"][0]
        return ChatResponse(
            content=choice["message"]["content"],
            model=data["model"],
            usage=data.get("usage", {}),
        )

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        session = await self._get_session()
        payload = {
            "model": model or self.config.default_model,
            "messages": [m.to_dict() for m in messages],
            "stream": True,
            **kwargs,
        }

        try:
            async with session.post("chat/completions", json=payload) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise AIServiceError(
                        make_api_error_message(resp.status, text)
                    )

                buffer = ""
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        buffer = line[6:]
                        try:
                            data = json.loads(buffer)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (
                            json.JSONDecodeError,
                            KeyError,
                            IndexError,
                        ):
                            continue
        except aiohttp.ClientError as e:
            raise AIServiceError(
                _("Connection failed - please check your network")
            ) from e

    async def list_models(self) -> List[str]:
        session = await self._get_session()
        try:
            async with session.get("models") as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise AIServiceError(
                        make_api_error_message(resp.status, text)
                    )
                data = await resp.json()
                return [m["id"] for m in data.get("data", [])]
        except aiohttp.ClientError as e:
            raise AIServiceError(
                _("Connection failed - please check your network")
            ) from e

    async def test_connection(self) -> Tuple[bool, str]:
        try:
            models = await self.list_models()
        except AIServiceError as e:
            return False, str(e)

        model_name = self.config.default_model
        if model_name and models and model_name not in models:
            return False, _(
                "Model '{model}' not found. Available: {available}"
            ).format(
                model=model_name,
                available=", ".join(sorted(models)[:10]),
            )

        if models:
            return True, f"Connected. {len(models)} models available."
        return True, "Connected (no models listed)."

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
