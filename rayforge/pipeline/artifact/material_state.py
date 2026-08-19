from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from raygeo.ops.material.state import MaterialState


logger = logging.getLogger(__name__)


class MaterialStateArtifactHandle(BaseArtifactHandle):
    """Handle to a stored :class:`MaterialStateArtifact`."""

    def __init__(
        self,
        stock_uid: str,
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: dict[str, Any] | None = None,
        **_kwargs,
    ):
        super().__init__(
            key=key,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            generation_id=generation_id,
            array_metadata=array_metadata,
        )
        self.stock_uid = stock_uid


class MaterialStateArtifact(BaseArtifact):
    """The folded material state of one stock, as a pipeline artifact.

    Wraps the raygeo :class:`~raygeo.ops.material.state.MaterialState`
    (voids, surface map, provenance) alongside the stock identity and
    generation id. Consumers ask for a projection ("give me voids and a
    heightmap for this stock"), so renderer code never assumes which
    profile produced the state.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        material_state: MaterialState,
        stock_uid: str,
        generation_id: int,
    ):
        super().__init__()
        self.material_state = material_state
        self.stock_uid = stock_uid
        self.generation_id = generation_id

    def build_handle(self, key: str) -> MaterialStateArtifactHandle:
        return MaterialStateArtifactHandle(
            key=key,
            handle_class_name=MaterialStateArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
            stock_uid=self.stock_uid,
        )
