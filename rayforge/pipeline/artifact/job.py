from __future__ import annotations
import json
import numpy as np
from typing import Optional, Dict, Any, Type, TYPE_CHECKING
from ...core.ops import Ops
from .base import BaseArtifact, VertexData
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ..encoder.base import MachineCodeOpMap


class JobArtifactHandle(BaseArtifactHandle):
    """A handle for a JobArtifact."""

    def __init__(
        self,
        time_estimate: Optional[float],
        distance: float,
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            shm_name=shm_name,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            array_metadata=array_metadata,
        )
        self.time_estimate = time_estimate
        self.distance = distance


class JobArtifact(BaseArtifact):
    """
    Represents a final job artifact containing G-code and operation data
    for machine execution.
    """

    def __init__(
        self,
        ops: Ops,
        distance: float,
        time_estimate: Optional[float] = None,
        machine_code_bytes: Optional[np.ndarray] = None,
        op_map_bytes: Optional[np.ndarray] = None,
        vertex_data: Optional[VertexData] = None,
    ):
        super().__init__()
        self.ops = ops
        self.distance = distance
        self.time_estimate = time_estimate
        self.machine_code_bytes: Optional[np.ndarray] = machine_code_bytes
        self.op_map_bytes: Optional[np.ndarray] = op_map_bytes
        self.vertex_data: Optional[VertexData] = vertex_data

        # Caching properties for deserialized data
        self._machine_code_str: Optional[str] = None
        self._op_map_obj: Optional["MachineCodeOpMap"] = None

    @property
    def machine_code(self) -> Optional[str]:
        """
        Lazily decodes and caches the G-code string from its byte array.
        """
        if (
            self._machine_code_str is None
            and self.machine_code_bytes is not None
        ):
            self._machine_code_str = self.machine_code_bytes.tobytes().decode(
                "utf-8"
            )
        return self._machine_code_str

    @property
    def op_map(self) -> Optional["MachineCodeOpMap"]:
        """
        Lazily decodes and caches the MachineCodeOpMap from its byte array.
        """
        from ..encoder.base import MachineCodeOpMap

        if self._op_map_obj is None and self.op_map_bytes is not None:
            map_str = self.op_map_bytes.tobytes().decode("utf-8")
            map_dict = json.loads(map_str)
            self._op_map_obj = MachineCodeOpMap(
                op_to_machine_code={
                    int(k): v
                    for k, v in map_dict["op_to_machine_code"].items()
                },
                machine_code_to_op={
                    int(k): v
                    for k, v in map_dict["machine_code_to_op"].items()
                },
            )
        return self._op_map_obj

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = {
            "ops": self.ops.to_dict(),
            "time_estimate": self.time_estimate,
            "distance": self.distance,
        }
        if self.vertex_data is not None:
            result["vertex_data"] = self.vertex_data.to_dict()
        if self.machine_code_bytes is not None:
            result["machine_code_bytes"] = self.machine_code_bytes.tolist()
        if self.op_map_bytes is not None:
            result["op_map_bytes"] = self.op_map_bytes.tolist()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobArtifact":
        """Creates an artifact from a dictionary."""
        ops = Ops.from_dict(data["ops"])
        common_args = {
            "ops": ops,
            "time_estimate": data.get("time_estimate"),
            "distance": data.get("distance", 0.0),
        }
        if "vertex_data" in data:
            common_args["vertex_data"] = VertexData.from_dict(
                data["vertex_data"]
            )
        if "machine_code_bytes" in data:
            common_args["machine_code_bytes"] = np.array(
                data["machine_code_bytes"], dtype=np.uint8
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
            time_estimate=self.time_estimate,
            distance=self.distance,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        arrays = self.ops.to_numpy_arrays()
        if self.machine_code_bytes is not None:
            arrays["machine_code_bytes"] = self.machine_code_bytes
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
        if not isinstance(handle, JobArtifactHandle):
            raise TypeError("JobArtifact requires a JobArtifactHandle")

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
            time_estimate=handle.time_estimate,
            distance=handle.distance,
            machine_code_bytes=arrays.get(
                "machine_code_bytes", np.empty(0, dtype=np.uint8)
            ).copy(),
            op_map_bytes=arrays.get(
                "op_map_bytes", np.empty(0, dtype=np.uint8)
            ).copy(),
            vertex_data=vertex_data,
        )
