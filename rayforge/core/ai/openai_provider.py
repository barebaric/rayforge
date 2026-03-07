import json
import logging
from typing import AsyncGenerator, List, Optional, Tuple

import aiohttp

from .provider import AIProvider, AIProviderConfig, ChatMessage, ChatResponse

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

        async with session.post("chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=text,
                )
            data = await resp.json()

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

        async with session.post("chat/completions", json=payload) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise aiohttp.ClientResponseError(
                    request_info=resp.request_info,
                    history=resp.history,
                    status=resp.status,
                    message=text,
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
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    async def list_models(self) -> List[str]:
        session = await self._get_session()
        try:
            async with session.get("models") as resp:
                if resp.status != 200:
                    logger.warning(f"Failed to list models: {resp.status}")
                    return []
                data = await resp.json()
                return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.warning(f"Error listing models: {e}")
            return []

    async def test_connection(self) -> Tuple[bool, str]:
        try:
            models = await self.list_models()
            if models:
                return True, f"Connected. {len(models)} models available."
            return True, "Connected (no models listed)."
        except aiohttp.ClientError as e:
            return False, f"Connection failed: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
