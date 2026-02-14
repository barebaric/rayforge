from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Type, Union
from dataclasses import dataclass, asdict
from .base import BaseArtifact
from .handle import BaseArtifactHandle


@dataclass
class RenderContext:
    """
    An immutable contract describing all parameters from the UI required to
    perform a render of a workpiece view.
    """

    pixels_per_mm: Tuple[float, float]
    show_travel_moves: bool
    margin_px: int
    color_set_dict: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the context to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RenderContext":
        """Deserializes a RenderContext from a dictionary."""
        return cls(**data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RenderContext):
            return False
        return (
            self.pixels_per_mm == other.pixels_per_mm
            and self.show_travel_moves == other.show_travel_moves
            and self.margin_px == other.margin_px
            and self._compare_color_sets(
                self.color_set_dict, other.color_set_dict
            )
        )

    def _compare_color_sets(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> bool:
        """Compare two color set dictionaries for equality."""
        if dict1.keys() != dict2.keys():
            return False
        for key in dict1:
            val1 = dict1[key]
            val2 = dict2.get(key)
            if isinstance(val1, dict) and isinstance(val2, dict):
                if not self._compare_color_sets(val1, val2):
                    return False
            elif val1 != val2:
                return False
        return True


class WorkPieceViewArtifactHandle(BaseArtifactHandle):
    """A handle for a WorkPieceViewArtifact."""

    def __init__(
        self,
        bbox_mm: Tuple[float, float, float, float],
        workpiece_size_mm: Tuple[float, float],
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        array_metadata: Union[Dict[str, Any], None] = None,
        **_kwargs,
    ):
        super().__init__(
            shm_name=shm_name,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            array_metadata=array_metadata,
        )
        self.bbox_mm = bbox_mm
        self.workpiece_size_mm = workpiece_size_mm


class WorkPieceViewArtifact(BaseArtifact):
    """
    An artifact containing a pre-rendered bitmap of a workpiece for fast
    display on the 2D canvas.
    """

    def __init__(
        self,
        bitmap_data: np.ndarray,
        bbox_mm: Tuple[float, float, float, float],
        workpiece_size_mm: Tuple[float, float],
    ):
        super().__init__()
        self.bitmap_data = bitmap_data
        self.bbox_mm = bbox_mm
        self.workpiece_size_mm = workpiece_size_mm

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> WorkPieceViewArtifactHandle:
        """Creates the appropriate, typed handle for this artifact."""
        return WorkPieceViewArtifactHandle(
            shm_name=shm_name,
            handle_class_name=WorkPieceViewArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            array_metadata=array_metadata,
            bbox_mm=self.bbox_mm,
            workpiece_size_mm=self.workpiece_size_mm,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of all NumPy arrays that need to be stored in
        shared memory for this artifact.
        """
        return {"bitmap_data": self.bitmap_data}

    @classmethod
    def from_storage(
        cls: Type[WorkPieceViewArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> WorkPieceViewArtifact:
        """
        Reconstructs an artifact instance from its handle and a dictionary of
        NumPy array views from shared memory.
        """
        if not isinstance(handle, WorkPieceViewArtifactHandle):
            raise TypeError(
                "WorkPieceViewArtifact requires a WorkPieceViewArtifactHandle"
            )

        return cls(
            bitmap_data=arrays["bitmap_data"].copy(),
            bbox_mm=handle.bbox_mm,
            workpiece_size_mm=handle.workpiece_size_mm,
        )
