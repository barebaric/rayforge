"""
Sections that addons contribute to the layer settings dialog.

A section adds the settings of one kind of layer to the dialog, below
each other with the built-in groups. Like the property providers, the
sections are registered by the addons and cleaned up when an addon
unloads.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from gi.repository import Adw

from ...core.layer import Layer

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class LayerSettingsSectionRegistry:
    """Registry for layer settings section classes."""

    def __init__(self):
        self._sections: list[type[LayerSettingsSection]] = []
        self._addon_map: dict[type[LayerSettingsSection], str] = {}

    def register(
        self, section_cls: type["LayerSettingsSection"], addon_name: str
    ) -> None:
        """
        Register a layer settings section class.

        Args:
            section_cls: The section class to register
            addon_name: The addon name for cleanup during unload
        """
        if section_cls not in self._sections:
            self._sections.append(section_cls)
            if addon_name:
                self._addon_map[section_cls] = addon_name
            logger.debug(
                f"Registered layer settings section: {section_cls.__name__}"
            )

    def unregister(self, section_cls: type["LayerSettingsSection"]) -> bool:
        """
        Unregister a layer settings section class.

        Returns True if the section was found and removed.
        """
        if section_cls in self._sections:
            self._sections.remove(section_cls)
            self._addon_map.pop(section_cls, None)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all sections registered by a specific addon.

        Args:
            addon_name: The name of the addon to clean up

        Returns:
            The number of sections unregistered
        """
        to_remove = [
            cls for cls, name in self._addon_map.items() if name == addon_name
        ]
        for cls in to_remove:
            self._sections.remove(cls)
            del self._addon_map[cls]
        if to_remove:
            logger.debug(
                f"Unregistered {len(to_remove)} layer settings sections "
                f"from addon '{addon_name}'"
            )
        return len(to_remove)

    def sections_for_layer(
        self, layer: Layer, editor: "DocEditor"
    ) -> list["LayerSettingsSection"]:
        """
        Instances of the registered sections that want to show settings
        for *layer*, in registration order.
        """
        found = []
        for section_cls in self._sections:
            section = section_cls()
            if section.can_handle(layer, editor):
                found.append(section)
        return found

    def all(self) -> list[type["LayerSettingsSection"]]:
        """Return all registered section classes."""
        return self._sections.copy()


layer_settings_section_registry = LayerSettingsSectionRegistry()


class LayerSettingsSection(ABC):
    """
    Defines the contract for a component that adds the settings of one
    kind of layer to the layer settings dialog.
    """

    @abstractmethod
    def can_handle(self, layer: Layer, editor: "DocEditor") -> bool:
        """
        Returns True if this section is applicable to *layer*.
        """
        ...

    @abstractmethod
    def create_widgets(
        self, layer: Layer, editor: "DocEditor"
    ) -> list[Adw.PreferencesGroup]:
        """
        Creates the widgets for this section. Each widget must be an
        :class:`Adw.PreferencesGroup`, because the dialog puts them on
        its preferences page. Called once per dialog; the dialog calls
        :meth:`close` when it is closed.
        """
        ...

    def close(self) -> None:
        """Releases signal connections when the dialog closes."""
