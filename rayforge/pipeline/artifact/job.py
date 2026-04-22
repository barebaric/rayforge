from __future__ import annotations
import json
import numpy as np
from typing import Optional, Dict, Any, Type, TYPE_CHECKING
from ...core.ops import Ops
from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ..encoder.base import MachineCodeOpMap, EncodedOutput


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


class JobArtifactHandle(BaseArtifactHandle):
    """A handle for a JobArtifact."""

    def __init__(
        self,
        time_estimate: Optional[float],
        distance: float,
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            shm_name=shm_name,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            generation_id=generation_id,
            array_metadata=array_metadata,
        )
        self.time_estimate = time_estimate
        self.distance = distance


class JobArtifact(BaseArtifact):
    """
    Represents a final job artifact containing G-code and operation data
    for machine execution.

    Coordinate conventions:
        ops: Raw assembled operations in world-space coordinates. No
            rotary mapping applied. Used as input to Machine.encode_ops()
            which handles the full transform pipeline (rotary mapping +
            world→machine + WCS + Z-flip) internally.
        mapped_ops: Same operations with rotary axis mapping applied
            (Y→degrees for rotary layers). Suitable for 3D preview and
            playback (scene compiler, OpPlayer). Not suitable for G-code
            encoding (lacks machine-coordinate transforms).
    """

    def __init__(
        self,
        ops: Ops,
        distance: float,
        generation_id: int,
        time_estimate: Optional[float] = None,
        encoded_output_bytes: Optional[np.ndarray] = None,
        mapped_ops: Optional[Ops] = None,
    ):
        super().__init__()
        self.ops = ops
        self.distance = distance
        self.generation_id = generation_id
        self.time_estimate = time_estimate
        self.encoded_output_bytes: Optional[np.ndarray] = encoded_output_bytes
        self.mapped_ops: Optional[Ops] = mapped_ops

        self._encoded_output: Optional["EncodedOutput"] = None

    @property
    def machine_code(self) -> Optional[str]:
        """
        Lazily decodes and caches the G-code string from encoded_output.
        """
        encoded = self.encoded_output
        return encoded.text if encoded else None

    @property
    def op_map(self) -> Optional["MachineCodeOpMap"]:
        """
        Lazily decodes and caches the MachineCodeOpMap from encoded_output.
        """
        encoded = self.encoded_output
        return encoded.op_map if encoded else None

    @property
    def encoded_output(self) -> Optional["EncodedOutput"]:
        """
        Lazily decodes and caches the full EncodedOutput from its byte array.
        This includes text, op_map, and driver_data (e.g., binary for Ruida).
        """
        from ..encoder.base import EncodedOutput

        if (
            self._encoded_output is None
            and self.encoded_output_bytes is not None
        ):
            json_str = self.encoded_output_bytes.tobytes().decode("utf-8")
            self._encoded_output = EncodedOutput.from_json(json_str)
        return self._encoded_output

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = {
            "ops": self.ops.to_dict(),
            "time_estimate": self.time_estimate,
            "distance": self.distance,
            "generation_id": self.generation_id,
        }
        if self.encoded_output_bytes is not None:
            result["encoded_output_bytes"] = self.encoded_output_bytes.tolist()
        if self.mapped_ops is not None:
            result["mapped_ops"] = self.mapped_ops.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobArtifact":
        """Creates an artifact from a dictionary."""
        ops = Ops.from_dict(data["ops"])
        common_args = {
            "ops": ops,
            "time_estimate": data.get("time_estimate"),
            "distance": data.get("distance", 0.0),
            "generation_id": data["generation_id"],
        }
        if "encoded_output_bytes" in data:
            common_args["encoded_output_bytes"] = np.array(
                data["encoded_output_bytes"], dtype=np.uint8
            )
        if "mapped_ops" in data:
            common_args["mapped_ops"] = Ops.from_dict(data["mapped_ops"])
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
            generation_id=self.generation_id,
            time_estimate=self.time_estimate,
            distance=self.distance,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        arrays = self.ops.to_numpy_arrays()
        if self.encoded_output_bytes is not None:
            arrays["encoded_output_bytes"] = self.encoded_output_bytes
        if self.mapped_ops is not None:
            mapped_json = json.dumps(
                self.mapped_ops.to_dict(), cls=_NumpyEncoder
            )
            arrays["mapped_ops_bytes"] = np.frombuffer(
                mapped_json.encode("utf-8"), dtype=np.uint8
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

        # Create a shallow copy of the arrays dictionary for each constructor.
        # This allows each constructor to safely mutate its copy (e.g., with
        # .pop()) without affecting others. The underlying numpy arrays are
        # still views into shared memory.
        arrays_copy = arrays.copy()
        ops = Ops.from_numpy_arrays(arrays_copy)

        mapped_ops = None
        mob = arrays.get("mapped_ops_bytes")
        if mob is not None and mob.size > 0:
            mapped_ops = Ops.from_dict(
                json.loads(mob.tobytes().decode("utf-8"))
            )

        return cls(
            ops=ops,
            time_estimate=handle.time_estimate,
            distance=handle.distance,
            generation_id=handle.generation_id,
            encoded_output_bytes=arrays.get(
                "encoded_output_bytes", np.empty(0, dtype=np.uint8)
            ).copy(),
            mapped_ops=mapped_ops,
        )
