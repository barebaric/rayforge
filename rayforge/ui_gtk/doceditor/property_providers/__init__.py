from .base import PropertyProvider, property_provider_registry
from .transform import TransformPropertyProvider
from .variable_text import VariableTextPropertyProvider
from .workpiece import TabsPropertyProvider, WorkpieceInfoProvider


def register_builtin_providers():
    """Register all built-in property providers."""
    property_provider_registry.register(TransformPropertyProvider, "")
    property_provider_registry.register(WorkpieceInfoProvider, "")
    property_provider_registry.register(TabsPropertyProvider, "")
    property_provider_registry.register(VariableTextPropertyProvider, "")


__all__ = [
    "PropertyProvider",
    "TabsPropertyProvider",
    "TransformPropertyProvider",
    "VariableTextPropertyProvider",
    "WorkpieceInfoProvider",
    "property_provider_registry",
    "register_builtin_providers",
]
