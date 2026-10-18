from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Protocol,
    runtime_checkable,
)

from blinker import Signal

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from ..image.base_renderer import Renderer
    from ..image.structures import FillRenderData


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
        params: dict[str, Any] | None = None,
        *,
        resolved_text_cache: dict | None = None,
    ) -> tuple["Geometry", list["FillRenderData"]]:
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

    def get_template_sources(self) -> list[str]:
        """
        The raw contents of all template strings this provider renders,
        e.g. the text of the sketch's text boxes.
        """
        ...

    def variables_used(self) -> set[str]:
        """
        The template variable names these templates read. Callers use
        this to know which variables this provider depends on; the
        provider itself decides what its templates reference.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the provider to a dictionary."""
        ...
