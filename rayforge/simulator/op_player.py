from typing import Optional

from ..core.ops import Ops
from ..core.ops.commands import MovingCommand
from .machine_state import MachineState


class OpPlayer:
    def __init__(self, ops: Ops):
        if not ops or ops.is_empty():
            raise ValueError("OpPlayer requires a non-empty Ops")
        self.ops = ops
        self.state = MachineState()
        self._current_index: int = -1

    @property
    def current_index(self) -> int:
        return self._current_index

    def seek(self, index: int):
        self.state = MachineState()
        self._current_index = -1
        self.advance_to(index)

    def advance_to(self, index: int):
        if index < self._current_index:
            raise ValueError(
                f"Cannot advance backwards: current="
                f"{self._current_index}, requested={index}. "
                f"Use seek() instead."
            )
        if index >= len(self.ops):
            raise IndexError(
                f"Index {index} out of range "
                f"(ops has {len(self.ops)} commands)"
            )
        for i in range(self._current_index + 1, index + 1):
            self.state.apply_command(self.ops.commands[i], i)
        self._current_index = index

    def seek_last_movement(self) -> Optional[int]:
        last = None
        for i, cmd in enumerate(self.ops):
            if isinstance(cmd, MovingCommand):
                last = i
        if last is not None:
            self.seek(last)
        return last
