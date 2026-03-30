import logging
from typing import Dict, List, Optional

from blinker import Signal

from .openai_provider import OpenAICompatibleProvider
from .provider import (
    AIProvider,
    AIProviderConfig,
    AIProviderType,
    ChatResponse,
)

logger = logging.getLogger(__name__)


class AIService:
    """
    Central service for AI operations.

    Manages multiple AI providers and exposes a simple interface for addons.
    Addons should use this service rather than configuring providers directly.
    """

    def __init__(self):
        self._providers: Dict[str, AIProvider] = {}
        self._configs: Dict[str, AIProviderConfig] = {}
        self._default_provider_id: Optional[str] = None
        self.changed = Signal()

    @property
    def providers(self) -> Dict[str, AIProviderConfig]:
        """Return a copy of provider configs."""
        return dict(self._configs)

    @property
    def default_provider_id(self) -> Optional[str]:
        """Return the ID of the default provider."""
        return self._default_provider_id

    @default_provider_id.setter
    def default_provider_id(self, value: Optional[str]):
        if value and value not in self._configs:
            raise ValueError(f"Unknown provider: {value}")
        self._default_provider_id = value
        self.changed.send(self)

    def add_provider(self, config: AIProviderConfig) -> AIProvider:
        """
        Register a new AI provider.

        Args:
            config: Provider configuration.

        Returns:
            The created provider instance.
        """
        if config.id in self._providers:
            raise ValueError(f"Provider already exists: {config.id}")

        provider = self._create_provider(config)
        self._providers[config.id] = provider
        self._configs[config.id] = config

        if self._default_provider_id is None:
            self._default_provider_id = config.id

        self.changed.send(self)
        logger.info(f"Added AI provider: {config.name} ({config.id})")
        return provider

    def update_provider(self, config: AIProviderConfig) -> AIProvider:
        """
        Update an existing provider's configuration.

        Args:
            config: New provider configuration.

        Returns:
            The recreated provider instance.
        """
        if config.id not in self._providers:
            raise ValueError(f"Provider not found: {config.id}")

        old_provider = self._providers[config.id]
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(old_provider.close())
            else:
                loop.run_until_complete(old_provider.close())
        except Exception:
            pass

        provider = self._create_provider(config)
        self._providers[config.id] = provider
        self._configs[config.id] = config
        self.changed.send(self)
        logger.info(f"Updated AI provider: {config.name} ({config.id})")
        return provider

    def remove_provider(self, provider_id: str):
        """
        Remove a provider.

        Args:
            provider_id: ID of the provider to remove.
        """
        if provider_id not in self._providers:
            raise ValueError(f"Provider not found: {provider_id}")

        provider = self._providers[provider_id]
        import asyncio

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(provider.close())
            else:
                loop.run_until_complete(provider.close())
        except Exception:
            pass

        del self._providers[provider_id]
        del self._configs[provider_id]

        if self._default_provider_id == provider_id:
            remaining = list(self._configs.keys())
            self._default_provider_id = remaining[0] if remaining else None

        self.changed.send(self)
        logger.info(f"Removed AI provider: {provider_id}")

    def _create_provider(self, config: AIProviderConfig) -> AIProvider:
        if config.provider_type == AIProviderType.OPENAI_COMPATIBLE:
            return OpenAICompatibleProvider(config)
        raise ValueError(f"Unknown provider type: {config.provider_type}")

    def get_provider(
        self, provider_id: Optional[str] = None
    ) -> Optional[AIProvider]:
        """
        Get a provider by ID, or the default provider.

        Args:
            provider_id: Specific provider ID, or None for default.

        Returns:
            Provider instance if found and enabled, else None.
        """
        pid = provider_id or self._default_provider_id
        if pid and pid in self._providers:
            config = self._configs[pid]
            if config.enabled:
                return self._providers[pid]
        return None

    def get_config(self, provider_id: str) -> Optional[AIProviderConfig]:
        """Get a provider configuration by ID."""
        return self._configs.get(provider_id)

    async def chat(
        self, messages: List, provider_id: Optional[str] = None, **kwargs
    ) -> Optional[ChatResponse]:
        """
        Send a chat request using the specified or default provider.

        Args:
            messages: List of ChatMessage objects.
            provider_id: Specific provider ID, or None for default.
            **kwargs: Additional arguments passed to the provider.

        Returns:
            ChatResponse if successful, None if no provider available.
        """
        provider = self.get_provider(provider_id)
        if provider is None:
            return None
        return await provider.chat(messages, **kwargs)

    async def chat_stream(
        self, messages: List, provider_id: Optional[str] = None, **kwargs
    ):
        """
        Stream a chat response using the specified or default provider.

        Args:
            messages: List of ChatMessage objects.
            provider_id: Specific provider ID, or None for default.
            **kwargs: Additional arguments passed to the provider.

        Yields:
            Content chunks as they arrive.
        """
        provider = self.get_provider(provider_id)
        if provider is None:
            return
        async for chunk in provider.chat_stream(messages, **kwargs):
            yield chunk

    def load_from_config(self, data: Dict):
        """
        Load providers from persisted configuration.

        Args:
            data: Dictionary with "providers" list and "default_provider".
        """
        providers_data = data.get("providers", [])
        for provider_data in providers_data:
            try:
                config = AIProviderConfig.from_dict(provider_data)
                self.add_provider(config)
            except Exception as e:
                logger.error(f"Failed to load provider: {e}")

        default_id = data.get("default_provider")
        if default_id and default_id in self._configs:
            self._default_provider_id = default_id

    def to_dict(self) -> Dict:
        """
        Serialize service state for persistence.

        Returns:
            Dictionary with providers and default_provider.
        """
        return {
            "providers": [cfg.to_dict() for cfg in self._configs.values()],
            "default_provider": self._default_provider_id,
        }

    async def close_all(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            try:
                await provider.close()
            except Exception as e:
                logger.warning(f"Error closing provider: {e}")
