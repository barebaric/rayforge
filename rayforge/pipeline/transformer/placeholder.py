from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Optional, List

from .base import OpsTransformer

if TYPE_CHECKING:
    from ...core.geo import Geometry
    from ...core.ops import Ops
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.progress import ProgressContext


class PlaceholderTransformer(OpsTransformer):
    """
    Transformer that preserves configuration for unknown transformer types.

    This is used when a document contains a step whose transformer type is
    not available (e.g., because the addon that provides it is not installed).
    The placeholder preserves the original configuration so the document can
    be saved without data loss.
    """

    def __init__(self, original_name: str, config: dict):
        super().__init__()
        self._original_name = original_name
        self._config = config.copy()
        self._label = _("Missing: {}").format(original_name)

    @property
    def original_name(self) -> str:
        """Returns the original transformer name that was not found."""
        return self._original_name

    @property
    def label(self) -> str:
        """A short label for the transformation, used in UI."""
        return self._label

    @property
    def description(self) -> str:
        """A brief one-line description of the transformation."""
        return _("This transformer is not available.")

    def to_dict(self) -> dict:
        """Returns the preserved original configuration."""
        return self._config.copy()

    def run(
        self,
        ops: "Ops",
        workpiece: Optional["WorkPiece"] = None,
        context: Optional["ProgressContext"] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
    ) -> None:
        pass

    @classmethod
    def from_dict(cls, data: dict) -> "PlaceholderTransformer":
        original_name = data.get("name", "Unknown")
        return cls(original_name, data)
