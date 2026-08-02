"""Registry of resolvers that turn driver-reported features into
domain capabilities.

Drivers report plain :class:`DriverFeatures` data. Addons register
resolvers here (via the ``register_driver_capabilities`` hook) that
interpret those features into concrete, domain-specific capabilities
(e.g. the laser addon turns ``DriverFeatures.pwm`` into a
``PWMCapability``). The machine then resolves a head's effective
capabilities through this registry.
"""

from typing import TYPE_CHECKING, Callable, List, Tuple

from .capability import Capability

if TYPE_CHECKING:
    from ..machine.driver.driver import DriverFeatures

DriverFeatureResolver = Callable[["DriverFeatures"], Tuple[Capability, ...]]


class DriverCapabilityRegistry:
    """Collects ``DriverFeatures`` resolvers contributed by addons."""

    def __init__(self) -> None:
        self._resolvers: List[DriverFeatureResolver] = []

    def register(self, resolver: DriverFeatureResolver) -> None:
        """Register a callable that maps DriverFeatures to capabilities."""
        if resolver not in self._resolvers:
            self._resolvers.append(resolver)

    def resolve(self, features: "DriverFeatures") -> Tuple[Capability, ...]:
        """Resolve driver features into domain capabilities."""
        caps: List[Capability] = []
        for resolver in self._resolvers:
            caps.extend(resolver(features))
        return tuple(caps)


driver_capability_registry = DriverCapabilityRegistry()
