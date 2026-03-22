from .base import PropertyProvider, property_provider_registry
from .transform import TransformPropertyProvider
from .workpiece import WorkpieceInfoProvider, TabsPropertyProvider


def register_builtin_providers():
    """Register all built-in property providers."""
    property_provider_registry.register(TransformPropertyProvider)
    property_provider_registry.register(WorkpieceInfoProvider)
    property_provider_registry.register(TabsPropertyProvider)


__all__ = [
    "PropertyProvider",
    "property_provider_registry",
    "register_builtin_providers",
    "TransformPropertyProvider",
    "WorkpieceInfoProvider",
    "TabsPropertyProvider",
]
