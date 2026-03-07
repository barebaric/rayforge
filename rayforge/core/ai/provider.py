from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Dict, List, Optional, Tuple


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
