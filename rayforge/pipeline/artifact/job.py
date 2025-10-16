from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Type
from ...core.ops import Ops
from ..coord import CoordinateSystem
from ..encoder.gcode import GcodeOpMap
from .base import BaseArtifact, VertexData
from .handle import BaseArtifactHandle


@dataclass
class JobArtifactHandle(BaseArtifactHandle):
    """A handle for a JobArtifact. Currently has no extra metadata."""

    pass


class JobArtifact(BaseArtifact):
    """
    Represents a final job artifact containing G-code and operation data
    for machine execution.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        source_dimensions: Optional[Tuple[float, float]] = None,
        time_estimate: Optional[float] = None,
        gcode_bytes: Optional[np.ndarray] = None,
        op_map_bytes: Optional[np.ndarray] = None,
        vertex_data: Optional[VertexData] = None,
    ):
        super().__init__(
            ops=ops,
            is_scalable=is_scalable,
            source_coordinate_system=source_coordinate_system,
            source_dimensions=source_dimensions,
            time_estimate=time_estimate,
        )
        self.gcode_bytes: Optional[np.ndarray] = gcode_bytes
        self.op_map_bytes: Optional[np.ndarray] = op_map_bytes
        self.vertex_data: Optional[VertexData] = vertex_data

        # Caching properties for deserialized data
        self._gcode_str: Optional[str] = None
        self._op_map_obj: Optional[GcodeOpMap] = None

    @property
    def gcode(self) -> Optional[str]:
        """
        Lazily decodes and caches the G-code string from its byte array.
        """
        if self._gcode_str is None and self.gcode_bytes is not None:
            self._gcode_str = self.gcode_bytes.tobytes().decode("utf-8")
        return self._gcode_str

    @property
    def op_map(self) -> Optional[GcodeOpMap]:
        """
        Lazily decodes and caches the GcodeOpMap from its byte array.
        """
        if self._op_map_obj is None and self.op_map_bytes is not None:
            map_str = self.op_map_bytes.tobytes().decode("utf-8")
            map_dict = json.loads(map_str)
            self._op_map_obj = GcodeOpMap(
                op_to_gcode={
                    int(k): v for k, v in map_dict["op_to_gcode"].items()
                },
                gcode_to_op={
                    int(k): v for k, v in map_dict["gcode_to_op"].items()
                },
            )
        return self._op_map_obj

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = super().to_dict()
        if self.vertex_data is not None:
            result["vertex_data"] = self.vertex_data.to_dict()
        if self.gcode_bytes is not None:
            result["gcode_bytes"] = self.gcode_bytes.tolist()
        if self.op_map_bytes is not None:
            result["op_map_bytes"] = self.op_map_bytes.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobArtifact":
        """Creates an artifact from a dictionary."""
        ops = Ops.from_dict(data["ops"])
        common_args = {
            "ops": ops,
            "is_scalable": data["is_scalable"],
            "source_coordinate_system": CoordinateSystem[
                data["source_coordinate_system"]
            ],
            "source_dimensions": data.get("source_dimensions"),
            "time_estimate": data.get("time_estimate"),
        }
        if "vertex_data" in data:
            common_args["vertex_data"] = VertexData.from_dict(
                data["vertex_data"]
            )
        if "gcode_bytes" in data:
            common_args["gcode_bytes"] = np.array(
                data["gcode_bytes"], dtype=np.uint8
            )
        if "op_map_bytes" in data:
            common_args["op_map_bytes"] = np.array(
                data["op_map_bytes"], dtype=np.uint8
            )
        return cls(**common_args)

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> JobArtifactHandle:
        return JobArtifactHandle(
            shm_name=shm_name,
            handle_class_name=JobArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            time_estimate=self.time_estimate,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        arrays = self.ops.to_numpy_arrays()
        if self.gcode_bytes is not None:
            arrays["gcode_bytes"] = self.gcode_bytes
        if self.op_map_bytes is not None:
            arrays["op_map_bytes"] = self.op_map_bytes
        if self.vertex_data is not None:
            arrays["powered_vertices"] = self.vertex_data.powered_vertices
            arrays["powered_colors"] = self.vertex_data.powered_colors
            arrays["travel_vertices"] = self.vertex_data.travel_vertices
            arrays["zero_power_vertices"] = (
                self.vertex_data.zero_power_vertices
            )
        return arrays

    @classmethod
    def from_storage(
        cls: Type[JobArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> JobArtifact:
        ops = Ops.from_numpy_arrays(arrays)
        vertex_data = None
        if all(
            key in arrays
            for key in [
                "powered_vertices",
                "powered_colors",
                "travel_vertices",
                "zero_power_vertices",
            ]
        ):
            vertex_data = VertexData(
                powered_vertices=arrays["powered_vertices"].copy(),
                powered_colors=arrays["powered_colors"].copy(),
                travel_vertices=arrays["travel_vertices"].copy(),
                zero_power_vertices=arrays["zero_power_vertices"].copy(),
            )
        return cls(
            ops=ops,
            is_scalable=handle.is_scalable,
            source_coordinate_system=CoordinateSystem[
                handle.source_coordinate_system_name
            ],
            source_dimensions=handle.source_dimensions,
            time_estimate=handle.time_estimate,
            gcode_bytes=arrays.get(
                "gcode_bytes", np.empty(0, dtype=np.uint8)
            ).copy(),
            op_map_bytes=arrays.get(
                "op_map_bytes", np.empty(0, dtype=np.uint8)
            ).copy(),
            vertex_data=vertex_data,
        )
