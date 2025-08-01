from __future__ import annotations
import logging
import uuid
from abc import ABC
from typing import List, Optional, TYPE_CHECKING, Callable
from blinker import Signal
from ..config import task_mgr
from ..modifier import Modifier, MakeTransparent, ToGrayscale
from ..opsproducer import OpsProducer, OutlineTracer, EdgeTracer, Rasterizer
from ..opstransformer import OpsTransformer, Optimize, Smooth
from .laser import Laser

if TYPE_CHECKING:
    from .doc import Doc
    from .workpiece import WorkPiece
    from .workflow import Workflow
    from ..tasker.task import Task
    from .ops import Ops


logger = logging.getLogger(__name__)

MAX_VECTOR_TRACE_PIXELS = 16 * 1024 * 1024
DEBOUNCE_DELAY_MS = 250  # Delay in milliseconds for ops regeneration


class Step(ABC):
    """
    A set of modifiers and an OpsProducer that operate on WorkPieces.

    A Step is a stateless configuration object that defines a single
    operation (e.g., outline, engrave) to be performed.
    """

    typelabel: str

    def __init__(
        self,
        workflow: "Workflow",
        opsproducer: OpsProducer,
        laser: Laser,
        max_cut_speed: int,
        max_travel_speed: int,
        name: Optional[str] = None,
    ):
        if not self.typelabel:
            raise AttributeError("Subclass must set a typelabel attribute.")

        self.workflow: Optional["Workflow"] = workflow
        self.uid = str(uuid.uuid4())
        self.name = name or self.typelabel
        self.visible = True
        self.modifiers = [
            MakeTransparent(),
            ToGrayscale(),
        ]
        self._modifier_ref_for_pyreverse: Modifier
        self.opsproducer = opsproducer
        self._opstransformers: List[OpsTransformer] = []

        self.passes: int = 1
        self.pixels_per_mm = 50, 50

        # Specific signals for different types of changes
        self.changed = Signal()
        self.visibility_changed = Signal()

        self.ops_generation_starting = Signal()
        self.ops_chunk_available = Signal()
        self.ops_generation_finished = Signal()

        self.laser = laser
        self.set_laser(laser)

        self.power = self.laser.max_power
        self.cut_speed = max_cut_speed
        self.travel_speed = max_travel_speed
        self.air_assist = False

    @property
    def doc(self) -> Optional["Doc"]:
        """The parent Doc object, accessed via the Workflow."""
        return self.workflow.doc if self.workflow else None

    @property
    def opstransformers(self) -> List[OpsTransformer]:
        return self._opstransformers

    @opstransformers.setter
    def opstransformers(self, transformers: List[OpsTransformer]):
        self._disconnect_transformer_signals()
        self._opstransformers = transformers
        self._connect_transformer_signals()
        self.changed.send(self)

    def _on_transformer_changed(self, sender, **kwargs):
        """
        Handles data-changing signals from child transformers.
        Bubbles up the notification by firing this step's own `changed` signal.
        """
        logger.debug(
            f"Step '{self.name}': Notified of model change from "
            f"transformer '{sender.label}'. Firing own changed signal."
        )
        self.changed.send(self)

    def _connect_transformer_signals(self):
        """Connects the step's handlers to its transformers' signals."""
        for transformer in self._opstransformers:
            transformer.changed.connect(self._on_transformer_changed)

    def _disconnect_transformer_signals(self):
        """Disconnects the step's handlers from its transformers' signals."""
        for transformer in self._opstransformers:
            try:
                transformer.changed.disconnect(self._on_transformer_changed)
            except TypeError:
                pass  # Signal was not connected

    def _on_task_event_received(
        self, task: "Task", event_name: str, data: dict
    ):
        """A stable instance method to handle events from tasks."""
        if event_name != "ops_chunk" or not self.workflow:
            return

        logger.debug(
            f"Step '{self.name}': _on_task_event_received caught "
            f"'ops_chunk' from Task {task.key}."
        )

        step_uid, workpiece_uid = task.key

        workpiece = self.workflow.layer._get_workpiece_by_uid(
            workpiece_uid
        )
        if not workpiece:
            logger.warning(
                f"Step '{self.name}': Received chunk for deleted "
                f"workpiece {workpiece_uid}. Ignoring."
            )
            return

        chunk = data.get("chunk")
        generation_id = data.get("generation_id")
        if not chunk or generation_id is None:
            return

        self.ops_chunk_available.send(
            self,
            workpiece=workpiece,
            chunk=chunk,
            generation_id=generation_id,
        )

    def start_generation_task(
        self,
        workpiece: "WorkPiece",
        generation_id: int,
        when_done_callback: Callable,
    ) -> "Task":
        """
        Starts the asynchronous task to generate operations for a workpiece.
        """
        from .steprunner import run_step_in_subprocess

        settings = {
            "power": self.power,
            "cut_speed": self.cut_speed,
            "travel_speed": self.travel_speed,
            "air_assist": self.air_assist,
            "pixels_per_mm": self.pixels_per_mm,
        }

        # The task key MUST be stable to allow cancellation.
        # It should not include the generation_id.
        task_mgr_key = self.uid, workpiece.uid

        # This is the stable, race-free pattern.
        # We pass our stable instance method directly to the TaskManager.
        task = task_mgr.run_process(
            run_step_in_subprocess,
            # Pass generation_id as a regular argument to the subprocess
            # function
            workpiece.to_dict(),
            self.opsproducer.to_dict(),
            [m.to_dict() for m in self.modifiers],
            [o.to_dict() for o in self.opstransformers],
            self.laser.to_dict(),
            settings,
            generation_id,
            key=task_mgr_key,
            when_done=when_done_callback,
            when_event=self._on_task_event_received,
        )

        return task

    def get_ops(self, workpiece: "WorkPiece") -> Optional["Ops"]:
        """
        Retrieves the final, cached Ops for a workpiece by delegating
        the call to the parent Layer.
        """
        if not self.workflow:
            logger.warning(
                f"Cannot get_ops for Step '{self.name}': "
                "no parent workflow."
            )
            return None
        return self.workflow.layer.get_ops(self, workpiece)

    def set_passes(self, passes: bool = True):
        self.passes = int(passes)
        self.changed.send(self)

    def set_visible(self, visible: bool = True):
        self.visible = visible
        self.visibility_changed.send(self)

    def set_laser(self, laser: Laser):
        if laser == self.laser:
            return
        self.laser = laser
        self.changed.send(self)

    def set_power(self, power: int):
        self.power = power
        self.changed.send(self)

    def set_cut_speed(self, speed: int):
        self.cut_speed = int(speed)
        self.changed.send(self)

    def set_travel_speed(self, speed: int):
        self.travel_speed = int(speed)
        self.changed.send(self)

    def set_air_assist(self, enabled: bool):
        self.air_assist = bool(enabled)
        self.changed.send(self)

    def get_summary(self) -> str:
        power = int(self.power / self.laser.max_power * 100)
        speed = int(self.cut_speed)
        return f"{power}% power, {speed} mm/min"

    def can_scale(self) -> bool:
        return self.opsproducer.can_scale()

    def dump(self, indent: int = 0):
        print("  " * indent, self.name)


class Outline(Step):
    typelabel = _("External Outline")

    def __init__(
        self, *, workflow: "Workflow", name: Optional[str] = None, **kwargs
    ):
        super().__init__(
            workflow=workflow, opsproducer=OutlineTracer(), name=name, **kwargs
        )
        self.opstransformers = [
            Smooth(enabled=False, amount=20),
            Optimize(enabled=True),
        ]


class Contour(Step):
    typelabel = _("Contour")

    def __init__(
        self, *, workflow: "Workflow", name: Optional[str] = None, **kwargs
    ):
        super().__init__(
            workflow=workflow, opsproducer=EdgeTracer(), name=name, **kwargs
        )
        self.opstransformers = [
            Smooth(enabled=False, amount=20),
            Optimize(enabled=True),
        ]


class Rasterize(Step):
    typelabel = _("Raster Engrave")

    def __init__(
        self, *, workflow: "Workflow", name: Optional[str] = None, **kwargs
    ):
        super().__init__(
            workflow=workflow, opsproducer=Rasterizer(), name=name, **kwargs
        )
        self.opstransformers = [
            Optimize(enabled=True),
        ]
