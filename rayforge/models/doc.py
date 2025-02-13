from typing import List
from blinker import Signal
from .workpiece import WorkPiece
from .workplan import WorkPlan


class Doc:
    """
    Represents a loaded Rayforge document.
    """
    workpieces: List[WorkPiece]
    workplan: WorkPlan

    def __init__(self):
        self.workpieces = []
        self.workplan = WorkPlan("Default plan")
        self.changed = Signal()
        self.workplan.changed.connect(self.changed.send)

    def __iter__(self):
        return iter(self.workpieces)

    def add_workpiece(self, workpiece):
        self.workpieces.append(workpiece)

    def remove_workpiece(self, workpiece):
        if workpiece in self.workpieces:
            self.workpieces.remove(workpiece)

    def has_workpiece(self):
        return bool(self.workpieces)
