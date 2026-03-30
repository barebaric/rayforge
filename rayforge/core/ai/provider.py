import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from gettext import gettext as _
from typing import AsyncGenerator, Dict, List, Optional, Tuple


class AIServiceError(Exception):
    """Raised when AI service encounters an error."""

    pass


HTTP_STATUS_MESSAGES = {
    400: _("Bad request"),
    401: _("Authentication failed - please check your API key"),
    403: _("Access forbidden - please check your API key permissions"),
    404: _("API endpoint not found - please check the base URL"),
    429: _("Rate limited - please wait and try again"),
    500: _("Server error - please try again later"),
    502: _("Server error - please try again later"),
    503: _("Service unavailable - please try again later"),
}


def extract_api_error(body: str) -> Optional[str]:
    """Extract a human-readable error from a JSON API response body."""
    try:
        data = json.loads(body)
        if isinstance(data, list):
            data = data[0] if data else {}
        error = data.get("error", data)
        if isinstance(error, dict):
            return error.get("message") or error.get("status")
        return str(error) if error else None
    except (json.JSONDecodeError, AttributeError):
        return None


def make_api_error_message(status: int, body: str) -> str:
    """Build a user-friendly error message from an HTTP status and body."""
    generic = HTTP_STATUS_MESSAGES.get(
        status,
        _("Server returned error {code}").format(code=status),
    )
    detail = extract_api_error(body) if body else None
    if detail:
        return f"{generic}: {detail}"
    return generic


class AIProviderType(Enum):
    OPENAI_COMPATIBLE = "openai_compatible"


@dataclass
class AIProviderConfig:
    id: str
    name: str
    provider_type: AIProviderType
    api_key: str
    base_url: str
    default_model: str
    enabled: bool = True

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "provider_type": self.provider_type.value,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "default_model": self.default_model,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "AIProviderConfig":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            provider_type=AIProviderType(str(data.get("provider_type", ""))),
            api_key=str(data.get("api_key", "")),
            base_url=str(data.get("base_url", "")),
            default_model=str(data.get("default_model", "")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)


class AIProvider(ABC):
    """Base interface for AI providers."""

    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send a chat completion request."""
        pass

    @abstractmethod
    def chat_stream(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat completion response.

        This is an async generator that yields content chunks.
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models."""
        pass

    @abstractmethod
    async def test_connection(self) -> Tuple[bool, str]:
        """
        Test if the provider is reachable and configured.

        Returns:
            Tuple of (success, message) where message describes the result.
        """
        pass

    @abstractmethod
    async def close(self):
        """Close any open connections."""
        pass
