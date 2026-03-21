from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .geo import Geometry


class IGeometryProvider:
    """Protocol for assets that can provide geometry for a Workpiece."""

    @property
    def provider_type_name(self) -> str:
        """The type name for geometry provider identification."""
        ...

    def get_geometry(
        self, params: Optional[Dict[str, Any]] = None
    ) -> Tuple["Geometry", List["Geometry"]]:
        """
        Generate geometry with optional parameter overrides.

        Args:
            params: Optional dictionary of parameter values to override
                    the provider's default values.

        Returns:
            A tuple of (stroke_geometry, fill_geometries).
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the provider to a dictionary."""
        ...
