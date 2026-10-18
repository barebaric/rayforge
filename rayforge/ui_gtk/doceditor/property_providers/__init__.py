from .base import PropertyProvider, property_provider_registry
from .transform import TransformPropertyProvider
from .job_counter import JobCounterPropertyProvider
from .workpiece import TabsPropertyProvider, WorkpieceInfoProvider


def register_builtin_providers():
    """Register all built-in property providers."""
    property_provider_registry.register(TransformPropertyProvider, "")
    property_provider_registry.register(WorkpieceInfoProvider, "")
    property_provider_registry.register(TabsPropertyProvider, "")
    property_provider_registry.register(JobCounterPropertyProvider, "")


__all__ = [
    "PropertyProvider",
    "TabsPropertyProvider",
    "TransformPropertyProvider",
    "JobCounterPropertyProvider",
    "WorkpieceInfoProvider",
    "property_provider_registry",
    "register_builtin_providers",
]
