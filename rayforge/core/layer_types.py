"""
Layer types and the registry that describes them.

A layer type marks a layer as belonging to a special kind of content,
such as a generated calibration test. The type is stored on the layer
(:attr:`rayforge.core.layer.Layer.layer_type`) and saved with the
project; addons register what their layer types mean here. Layers with
an unknown or empty type behave like ordinary layers.
"""

import logging
from dataclasses import dataclass
from gettext import gettext as _
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .layer import Layer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerType:
    """Describes one kind of layer."""

    id: str
    """The value stored in :attr:`rayforge.core.layer.Layer.layer_type`."""

    title: str
    """The user-facing name of the kind of layer."""

    icon_name: str = ""
    """The icon shown next to the layer, when it has one."""

    locked: bool = False
    """
    Whether other workpieces must not be placed in the layer, because
    its content is managed by whatever generated it.
    """


class LayerTypeRegistry:
    """Registry of the layer types the addons provide."""

    def __init__(self):
        self._types: dict[str, LayerType] = {}
        self._addon_map: dict[str, str] = {}

    def register(self, layer_type: LayerType, addon_name: str) -> None:
        """
        Register a layer type.

        Args:
            layer_type: The layer type to register
            addon_name: The addon name for cleanup during unload
        """
        self._types[layer_type.id] = layer_type
        if addon_name:
            self._addon_map[layer_type.id] = addon_name
        logger.debug(f"Registered layer type: {layer_type.id}")

    def get(self, type_id: str) -> LayerType | None:
        """The layer type with *type_id*, or None when it is unknown."""
        return self._types.get(type_id)

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all layer types registered by a specific addon.

        Args:
            addon_name: The name of the addon to clean up

        Returns:
            The number of layer types unregistered
        """
        to_remove = [
            type_id
            for type_id, name in self._addon_map.items()
            if name == addon_name
        ]
        for type_id in to_remove:
            del self._types[type_id]
            del self._addon_map[type_id]
        if to_remove:
            logger.debug(
                f"Unregistered {len(to_remove)} layer types "
                f"from addon '{addon_name}'"
            )
        return len(to_remove)

    def all(self) -> list[LayerType]:
        """Return all registered layer types."""
        return list(self._types.values())


layer_type_registry = LayerTypeRegistry()


def layer_is_locked(layer: "Layer") -> bool:
    """Whether other workpieces must not be placed in *layer*."""
    layer_type = layer_type_registry.get(layer.layer_type)
    return layer_type is not None and layer_type.locked


def locked_layer_notice(layer: "Layer") -> str:
    """Why nothing can be placed in *layer*, for a user-facing message."""
    layer_type = layer_type_registry.get(layer.layer_type)
    title = layer_type.title if layer_type else _("layer")
    return _("Workpieces cannot be added to the {title} layer.").format(
        title=title
    )
