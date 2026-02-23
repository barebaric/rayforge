"""
Localized field support for multilingual content.

This module provides the LocalizedField class which extends str to provide
transparent localization of text content.
"""

import os
import locale
from typing import Dict, Optional, Union

SUPPORTED_LANGUAGES = ["en", "de", "es", "fr", "pt", "uk", "zh_CN"]

LocalizedString = Union[str, Dict[str, str]]


def _get_context_language() -> Optional[str]:
    """
    Get the current language from the application context.

    This is a private helper that isolates the context dependency
    to this module only.

    Returns:
        Language code or None if context not initialized
    """
    from ...context import get_context

    try:
        ctx = get_context()
        return ctx.language
    except RuntimeError:
        return None


def normalize_language_code(code: str) -> Optional[str]:
    """
    Normalize a language code to our supported format.

    Args:
        code: Input language code (e.g., 'de', 'de-DE', 'zh-CN')

    Returns:
        Normalized code or None if not supported
    """
    if not code:
        return None

    normalized = code.replace("-", "_")

    if normalized in SUPPORTED_LANGUAGES:
        return normalized

    base = normalized.split("_")[0]
    if base in SUPPORTED_LANGUAGES:
        return base

    return None


def get_system_language() -> str:
    """
    Detect the system's preferred language.

    Returns:
        Language code (e.g., 'de', 'zh_CN') or 'en' as fallback
    """
    # Try LC_ALL, LC_CTYPE, LANG environment variables first
    for env_var in ("LC_ALL", "LC_CTYPE", "LANG"):
        env_lang = os.environ.get(env_var)
        if env_lang:
            # Extract language part (e.g., "de_DE.UTF-8" -> "de_DE")
            lang = env_lang.split(".")[0]
            normalized = normalize_language_code(lang)
            if normalized:
                return normalized

    # Fallback to locale module
    try:
        lang = locale.getlocale()[0]
        if lang:
            normalized = normalize_language_code(lang)
            if normalized:
                return normalized
    except (ValueError, TypeError):
        pass

    return "en"


class LocalizedField(str):
    """
    A string subclass that can have different values for different languages.

    This class extends str, so it behaves exactly like a string in all
    contexts (concatenation, formatting, comparison, etc.). The string
    value is automatically resolved to the current language from context.

    Example:
        >>> name = LocalizedField.from_yaml({"default": "Wood", "de": "Holz"})
        >>> # With context language = "de":
        >>> str(name)
        'Holz'
        >>> name.upper()
        'HOLZ'
        >>> f"Material: {name}"
        'Material: Holz'

    Attributes:
        translations: Dictionary mapping language codes to translations
    """

    __slots__ = ("_default", "_translations")

    def __new__(
        cls, default: str, translations: Optional[Dict[str, str]] = None
    ):
        """
        Create a new LocalizedField.

        The string value is resolved from context language at creation time.

        Args:
            default: The default value used when no translation is available
            translations: Dictionary mapping language codes to translations
        """
        language = _get_context_language()
        translations = translations or {}
        value = translations.get(language, default) if language else default

        instance = super().__new__(cls, value)
        instance._default = default
        instance._translations = translations
        return instance

    @property
    def default(self) -> str:
        """Get the default value."""
        return self._default

    @property
    def translations(self) -> Dict[str, str]:
        """Get all translations."""
        return self._translations.copy()

    @classmethod
    def from_yaml(cls, value: LocalizedString) -> "LocalizedField":
        """
        Create a LocalizedField from YAML data.

        Handles both simple strings and localized objects.

        Args:
            value: Either a simple string or a dict with 'default' and lang
                keys

        Returns:
            A new LocalizedField instance
        """
        if isinstance(value, str):
            return cls(default=value)

        if isinstance(value, dict):
            default = value.get("default", "")
            translations = {k: v for k, v in value.items() if k != "default"}
            return cls(default=default, translations=translations)

        return cls(default=str(value))

    def to_yaml(self) -> LocalizedString:
        """
        Convert to YAML-compatible format.

        Returns:
            Simple string if no translations, otherwise dict format
        """
        if not self._translations:
            return self._default
        return {"default": self._default, **self._translations}

    def get(self, language: Optional[str] = None) -> str:
        """
        Get the value for a specific language.

        Args:
            language: Language code, or None to use context language

        Returns:
            The localized string or default if not available
        """
        if language is None:
            language = _get_context_language()

        if language is None:
            return self._default

        return self._translations.get(language, self._default)

    def get_all_values(self) -> Dict[str, str]:
        """
        Get all available translations including default.

        Returns:
            Dictionary with 'default' key and all language codes
        """
        result = {"default": self._default}
        result.update(self._translations)
        return result

    def matches(self, query: str) -> bool:
        """
        Check if any translation contains the query string.

        Args:
            query: Search string (case-insensitive)

        Returns:
            True if query is found in default or any translation
        """
        query = query.lower()
        if query in self._default.lower():
            return True
        for translation in self._translations.values():
            if query in translation.lower():
                return True
        return False

    def __repr__(self) -> str:
        """Represent the field showing its structure."""
        if not self._translations:
            return f"LocalizedField({self._default!r})"
        return f"LocalizedField({self._default!r}, {self._translations!r})"
