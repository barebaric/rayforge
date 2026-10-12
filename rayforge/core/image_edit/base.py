from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pyvips

_KNOWN_KEYS = frozenset({"op", "params", "enabled"})


@dataclass(frozen=True)
class ImageEditStep:
    """
    One non-destructive image editing operation in a workpiece's edit
    stack. ``op`` is the registry name of an ImageEditOp; ``params``
    must be JSON-serializable. Keys this version does not know are kept
    in ``extra`` and written back, so a newer file survives a round trip.
    """

    op: str
    params: Mapping[str, Any] = field(default_factory=dict)
    enabled: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "op": self.op,
            "params": dict(self.params),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageEditStep:
        return cls(
            op=data["op"],
            params=dict(data.get("params", {})),
            enabled=data.get("enabled", True),
            extra={k: v for k, v in data.items() if k not in _KNOWN_KEYS},
        )


class ImageEditOp(ABC):
    """
    An image editing operation. Subclasses are registered with the
    ImageEditRegistry under ``NAME``, which is stored in documents and
    must never change.
    """

    NAME: ClassVar[str]
    LABEL: ClassVar[str]

    @classmethod
    @abstractmethod
    def apply(
        cls, image: pyvips.Image, params: Mapping[str, Any]
    ) -> pyvips.Image:
        """Returns the edited image; ``image`` is left untouched."""
