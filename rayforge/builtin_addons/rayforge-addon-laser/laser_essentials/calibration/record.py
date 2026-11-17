"""
The record of a generated calibration test, saved with the project.

It keeps the generator, its parameters and the uids of everything the
generator made, so the test can be rebuilt when a parameter changes. The
record is a hidden asset: it is saved in the ``.ryp`` like any asset but
does not show in the asset browser.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from blinker import Signal

FORMAT_VERSION = 1


@dataclass(frozen=True)
class CalibrationState:
    """The parameters of a test and the items generated from them."""

    params: dict[str, Any]
    workpiece_uids: tuple[str, ...] = ()
    step_uids: tuple[str, ...] = ()
    asset_uids: tuple[str, ...] = ()


@dataclass
class CalibrationRecord:
    """A hidden asset that describes one generated calibration test."""

    is_addable: ClassVar[bool] = False
    asset_type_name: ClassVar[str] = "calibration_test"
    display_icon_name: ClassVar[str] = "test-symbolic"
    is_reorderable: ClassVar[bool] = False
    is_draggable_to_canvas: ClassVar[bool] = False
    type_display_name: ClassVar[str] = "Calibration Test"
    can_edit: ClassVar[bool] = False
    add_action: ClassVar[str | None] = None
    activate_action: ClassVar[str | None] = None
    edit_item_action: ClassVar[str | None] = None

    generator_id: str
    layer_uid: str
    state: CalibrationState
    name: str = ""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    updated: Signal = field(
        default_factory=Signal, init=False, repr=False, compare=False
    )

    @property
    def hidden(self) -> bool:
        return True

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.state.params)

    def set_state(self, state: CalibrationState) -> None:
        self.state = state
        self.updated.send(self)

    def owns(self, uid: str) -> bool:
        return uid in self.state.workpiece_uids

    def get_thumbnail(self, size: int) -> bytes | None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "type": self.asset_type_name,
            "name": self.name,
            "version": FORMAT_VERSION,
            "generator_id": self.generator_id,
            "layer_uid": self.layer_uid,
            "params": _jsonable(self.state.params),
            "workpiece_uids": list(self.state.workpiece_uids),
            "step_uids": list(self.state.step_uids),
            "asset_uids": list(self.state.asset_uids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationRecord":
        state = CalibrationState(
            params=dict(data.get("params", {})),
            workpiece_uids=tuple(data.get("workpiece_uids", ())),
            step_uids=tuple(data.get("step_uids", ())),
            asset_uids=tuple(data.get("asset_uids", ())),
        )
        return cls(
            generator_id=data["generator_id"],
            layer_uid=data.get("layer_uid", ""),
            state=state,
            name=data.get("name", ""),
            uid=data.get("uid", str(uuid.uuid4())),
        )


def _jsonable(params: dict[str, Any]) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in params.items()
    }
