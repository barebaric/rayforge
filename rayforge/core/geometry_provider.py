from typing import (
    Dict,
    Any,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
    TYPE_CHECKING,
)
from blinker import Signal

if TYPE_CHECKING:
    from ..image.base_renderer import Renderer
    from ..image.structures import FillRenderData
    from .geo import Geometry


@runtime_checkable
class IGeometryProvider(Protocol):
    """Protocol for assets that can provide geometry for a Workpiece."""

    @property
    def uid(self) -> str:
        """The unique identifier of the provider asset."""
        ...

    @property
    def name(self) -> str:
        """The user-facing name of the provider asset."""
        ...

    @property
    def updated(self) -> "Signal":
        """Signal emitted when the provider's geometry changes."""
        ...

    @property
    def provider_type_name(self) -> str:
        """The type name for geometry provider identification."""
        ...

    @property
    def renderer(self) -> Optional["Renderer"]:
        """The renderer to use for rendering this provider's geometry."""
        ...

    def get_geometry(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        resolved_text_cache: Optional[Dict] = None,
    ) -> Tuple["Geometry", List["FillRenderData"]]:
        """
        Generate geometry with optional parameter overrides.

        Args:
            params: Optional dictionary of parameter values to override
                    the provider's default values.
            resolved_text_cache: Optional mutable dict that carries
                    resolved template text across calls so volatile
                    expressions like ``uuid4()`` stay consistent.

        Returns:
            A tuple of (stroke_geometry, fill_render_data).
        """
        ...

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the provider to a dictionary."""
        ...
