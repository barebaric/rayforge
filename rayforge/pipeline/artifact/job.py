from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from raygeo.ops import Ops

from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ..encoder.base import EncodedOutput, MachineCodeOpMap


class JobArtifactHandle(BaseArtifactHandle):
    def __init__(
        self,
        time_estimate: Optional[float],
        distance: float,
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            key=key,
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
        mapped_ops: Optional[Ops] = None,
        encoded_output: Optional["EncodedOutput"] = None,
    ):
        super().__init__()
        self.ops = ops
        self.distance = distance
        self.generation_id = generation_id
        self.time_estimate = time_estimate
        self.mapped_ops: Optional[Ops] = mapped_ops

        self._encoded_output: Optional["EncodedOutput"] = encoded_output

    @property
    def preview_ops(self) -> Optional[Ops]:
        """Returns the ops suitable for 3D preview/playback.

        Prefers the rotary-mapped ops (``mapped_ops``) when available,
        falling back to the raw assembled ``ops``.
        """
        return self.mapped_ops if self.mapped_ops is not None else self.ops

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
        """Returns the cached EncodedOutput, if any."""
        return self._encoded_output

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = {
            "ops": self.ops.to_dict(),
            "time_estimate": self.time_estimate,
            "distance": self.distance,
            "generation_id": self.generation_id,
        }
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
        if "mapped_ops" in data:
            common_args["mapped_ops"] = Ops.from_dict(data["mapped_ops"])
        return cls(**common_args)

    def build_handle(self, key: str) -> JobArtifactHandle:
        return JobArtifactHandle(
            key=key,
            handle_class_name=JobArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
            time_estimate=self.time_estimate,
            distance=self.distance,
        )
