from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass, field
import base64
import json
from ...core.ops import Ops


@dataclass
class MachineCodeOpMap:
    """
    A container for a bidirectional mapping between Ops command indices and
    Machine language (e.g. G-code) line numbers.

    Attributes:
        op_to_machine_code: Maps an Ops command index to a list of G-code line
                     numbers it generated. An empty list means the command
                     produced no G-code.
        machine_code_to_op: Maps a G-code line number back to the Ops command
                     index that generated it.
    """

    op_to_machine_code: Dict[int, List[int]] = field(default_factory=dict)
    machine_code_to_op: Dict[int, int] = field(default_factory=dict)


@dataclass
class EncodedOutput:
    """
    Base class for encoder output. Must be JSON-serializable
    via to_dict/from_dict.

    This provides a unified interface for all encoder outputs,
    allowing the UI
    to work with any encoder type without needing to know the specifics.

    Attributes:
        text: Human-readable machine code representation for UI display.
        op_map: Bidirectional mapping between ops indices and line numbers.
        driver_data: Optional driver-specific data (e.g., binary for Ruida).
                     Bytes values are stored as base64-encoded strings.
    """

    text: str
    op_map: MachineCodeOpMap
    driver_data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to a JSON-compatible dictionary.
        Bytes values in driver_data are converted to base64 strings.
        """
        serialized_driver_data = {}
        for key, value in self.driver_data.items():
            if isinstance(value, bytes):
                serialized_driver_data[key] = {
                    "__type__": "bytes",
                    "data": base64.b64encode(value).decode("ascii"),
                }
            else:
                serialized_driver_data[key] = value

        return {
            "text": self.text,
            "op_map": {
                "op_to_machine_code": {
                    str(k): v
                    for k, v in self.op_map.op_to_machine_code.items()
                },
                "machine_code_to_op": {
                    str(k): v
                    for k, v in self.op_map.machine_code_to_op.items()
                },
            },
            "driver_data": serialized_driver_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncodedOutput":
        """
        Deserialize from a dictionary.
        Base64-encoded bytes in driver_data are converted back to bytes.
        """
        op_map_data = data["op_map"]
        op_map = MachineCodeOpMap(
            op_to_machine_code={
                int(k): v for k, v in op_map_data["op_to_machine_code"].items()
            },
            machine_code_to_op={
                int(k): v for k, v in op_map_data["machine_code_to_op"].items()
            },
        )

        deserialized_driver_data = {}
        for key, value in data.get("driver_data", {}).items():
            if isinstance(value, dict) and value.get("__type__") == "bytes":
                deserialized_driver_data[key] = base64.b64decode(value["data"])
            else:
                deserialized_driver_data[key] = value

        return cls(
            text=data["text"],
            op_map=op_map,
            driver_data=deserialized_driver_data,
        )

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "EncodedOutput":
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(json_str))


class OpsEncoder(ABC):
    """
    Transforms an Ops object into something else.
    Examples:

    - Ops to image (a cairo surface)
    - Ops to a G-code string
    """

    @abstractmethod
    def encode(self, ops: Ops, *args, **kwargs) -> Any:
        pass
