from __future__ import annotations
from typing import List, Dict
from copy import deepcopy
from ..task import task_mgr, CancelledError
from ..config import config, getflag
from ..modifier import Modifier, MakeTransparent, ToGrayscale
from ..opsproducer import OpsProducer, OutlineTracer, EdgeTracer, Rasterizer
from ..opstransformer import OpsTransformer, Optimize, Smooth, ArcWeld
from .workpiece import WorkPiece
from .machine import Laser
from .ops import Ops
from blinker import Signal


DEBUG_OPTIMIZE = getflag('DEBUG_OPTIMIZE')
DEBUG_SMOOTH = getflag('DEBUG_SMOOTH')
DEBUG_ARCWELD = getflag('DEBUG_ARCWELD')


class WorkStep:
    """
    A WorkStep is a set of Modifiers that operate on a set of
    WorkPieces. It normally generates a Ops in the end, but
    may also include modifiers that manipulate the input image.
    """
    typelabel = None

    def __init__(self, opsproducer: OpsProducer, name=None):
        self.workplan: WorkPlan = None
        self.name: str = name or self.typelabel
        self.visible: bool = True
        self.modifiers: List[Modifier] = [
            MakeTransparent(),
            ToGrayscale(),
        ]
        self._modifier_ref_for_pyreverse: Modifier
        self.opsproducer: OpsProducer = opsproducer
        self.opstransformers: List[OpsTransformer] = []
        self._opstransformer_ref_for_pyreverse: OpsTransformer

        # Map WorkPieces to Ops and size
        self.workpiece_to_ops: Dict[WorkPiece, [Ops, [float, float]]] = {}
        self._workpiece_ref_for_pyreverse: WorkPiece
        self._ops_ref_for_pyreverse: Ops

        self.passes: int = 1
        self.pixels_per_mm = 25, 25
        self.laser: Laser = None

        self.changed = Signal()
        self.ops_changed: Signal = Signal()
        self.set_laser(config.machine.heads[0])

        self.power: int = self.laser.max_power
        self.cut_speed: int = config.machine.max_cut_speed
        self.travel_speed: int = config.machine.max_travel_speed
        self.air_assist: bool = False

        if DEBUG_OPTIMIZE:
            self.opstransformers.append(Optimize())
        if DEBUG_SMOOTH:
            self.opstransformers.append(Smooth())
        if DEBUG_ARCWELD:
            self.opstransformers.append(ArcWeld())

    def set_passes(self, passes=True):
        self.passes = int(passes)
        self.changed.send(self)

    def set_visible(self, visible=True):
        self.visible = visible
        self.changed.send(self)

    def set_laser(self, laser):
        if laser == self.laser:
            return
        if self.laser:
            self.laser.changed.disconnect(self._on_laser_changed)
        self.laser = laser
        laser.changed.connect(self._on_laser_changed)
        self.update_all_workpieces()
        self.changed.send(self)

    def _on_laser_changed(self, sender, **kwargs):
        self.update_all_workpieces()
        self.changed.send(self)

    def set_power(self, power):
        self.power = power
        self.update_all_workpieces()
        self.changed.send(self)

    def set_workpieces(self, workpieces: List[WorkPiece]):
        for workpiece in list(self.workpiece_to_ops.keys()):
            if workpiece in workpieces:
                continue
            workpiece.size_changed.disconnect(self._on_workpiece_size_changed)
            del self.workpiece_to_ops[workpiece]
        for workpiece in workpieces:
            self.add_workpiece(workpiece)
        self.changed.send(self)

    def add_workpiece(self, workpiece: WorkPiece):
        if workpiece in self.workpiece_to_ops:
            return
        self.workpiece_to_ops[workpiece] = None, None
        workpiece.size_changed.connect(self._on_workpiece_size_changed)
        self.update_workpiece(workpiece)
        self.changed.send(self)

    def remove_workpiece(self, workpiece: WorkPiece):
        workpiece.size_changed.disconnect(self._on_workpiece_size_changed)
        del self.workpiece_to_ops[workpiece]
        self.changed.send(self)

    def _on_workpiece_size_changed(self, workpiece):
        if not self.can_scale():
            self.update_workpiece(workpiece)

    def workpieces(self):
        return self.workpiece_to_ops.keys()

    def execute(self, workpiece) -> [Ops, [float, float]]:
        """
        workpiece: the input workpiece to generate Ops for.
        """
        if self.can_scale():
            # Since we are producing a vector as an output, and the
            # result is later scaled, the render size does not matter much
            # unless it is so small that rounding errors become relevant.
            # So we can choose a relatively small value for speed and memory
            # efficiency.
            size = 100, 100  # in mm
            surface, _ = workpiece.render(*self.pixels_per_mm,
                                           size=size,
                                           force=True)

            # There is no guarantee that the renderer was able to deliver
            # the size we asked for. Check the actual size.
            width, height = surface.get_width(), surface.get_height()
            width_mm = width / self.pixels_per_mm[0]
            height_mm = height / self.pixels_per_mm[1]
            size = width_mm, height_mm

            chunks = [(surface, (0, 0))]

        else:
            # Rendering a large work surface (say, 1 x 2 meters for example)
            # at the required pixel density is unmanagably large.
            # So we must render and process the surface in chunks.
            size = workpiece.size
            chunks = workpiece.render_chunk(*self.pixels_per_mm,
                                            size=size,
                                            force=True)

        ops = Ops()
        ops.set_power(self.power)
        ops.set_cut_speed(self.cut_speed)
        ops.set_travel_speed(self.travel_speed)
        ops.enable_air_assist(self.air_assist)

        for surface, (x_offset, y_offset) in chunks:
            # Apply bitmap modifiers.
            for modifier in self.modifiers:
                modifier.run(surface)

            # Produce an Ops object from the resulting surface.
            chunk_ops = self.opsproducer.run(
                config.machine,
                self.laser,
                surface,
                self.pixels_per_mm
            )

            y_offset = size[1] - (surface.get_height()+y_offset) \
                     / self.pixels_per_mm[1]
            chunk_ops.translate(x_offset/self.pixels_per_mm[0], y_offset)
            ops += chunk_ops
            surface.flush()  # Free memory after use

        # Apply Ops object transformations.
        for transformer in self.opstransformers:
            transformer.run(ops)

        ops.disable_air_assist()
        self.workpiece_to_ops[workpiece] = ops, size
        return ops, size

    async def execute_async(self, workpiece: WorkPiece) -> [
            WorkPiece, Ops, [float, float]]:
        ops, size = self.execute(workpiece)
        return workpiece, ops, size

    def update_workpiece(self, workpiece):
        key = id(self), id(workpiece)
        task_mgr.add_coroutine(
            self.execute_async(workpiece),
            when_done=self._on_ops_created,
            key=key
        )

    def update_all_workpieces(self):
        for workpiece in self.workpiece_to_ops.keys():
            self.update_workpiece(workpiece)

    def _on_ops_created(self, task):
        try:
            workpiece, ops, size = task.result()
        except CancelledError:
            return
        self.ops_changed.send(self, workpiece=workpiece)

    def get_ops(self, workpiece):
        """
        Returns Ops for the given workpiece, scaled to the size of
        the workpiece.
        Returns None if no Ops were made yet.
        """
        ops, size = self.workpiece_to_ops.get(workpiece, (None, None))
        if ops is None:
            return None
        orig_width_mm, orig_height_mm = size
        width_mm, height_mm = workpiece.size
        ops = deepcopy(ops)
        ops.scale(width_mm/orig_width_mm, height_mm/orig_height_mm)
        return ops

    def get_summary(self):
        power = int(self.power/self.laser.max_power*100)
        speed = int(self.cut_speed)
        return f"{power}% power, {speed} mm/min"

    def can_scale(self):
        return self.opsproducer.can_scale()

    def dump(self, indent=0):
        print("  "*indent, self.name)
        for workpiece in self.workpieces():
            workpiece.dump(1)


class Outline(WorkStep):
    typelabel = "External Outline"

    def __init__(self, name=None, **kwargs):
        super().__init__(OutlineTracer(), name, **kwargs)


class Contour(WorkStep):
    typelabel = "Contour"

    def __init__(self, name=None, **kwargs):
        super().__init__(EdgeTracer(), name, **kwargs)


class Rasterize(WorkStep):
    typelabel = "Raster Engrave"

    def __init__(self, name=None, **kwargs):
        super().__init__(Rasterizer(), name, **kwargs)


class WorkPlan:
    """
    Represents a sequence of worksteps.
    """
    def __init__(self, doc, name):
        self.doc = doc
        self.name: str = name
        self.worksteps: List[WorkStep] = []
        self._workstep_ref_for_pyreverse: WorkStep
        self.changed = Signal()
        self.add_workstep(Contour())

    def __iter__(self):
        return iter(self.worksteps)

    def set_workpieces(self, workpieces):
        for step in self.worksteps:
            step.set_workpieces(workpieces)

    def add_workstep(self, step):
        step.workplan = self
        self.worksteps.append(step)
        step.set_workpieces(self.doc.workpieces)
        self.changed.send(self)

    def remove_workstep(self, workstep):
        self.worksteps.remove(workstep)
        workstep.workplan = None
        self.changed.send(self)

    def set_worksteps(self, worksteps):
        """
        Replace all worksteps.
        """
        self.worksteps = worksteps
        for step in worksteps:
            step.workplan = self
        self.changed.send(self)

    def has_steps(self):
        return len(self.worksteps) > 0

    def execute(self, optimize=True):
        ops = Ops()
        for step in self.worksteps:
            for workpiece in step.workpieces():
                step.execute(workpiece)
                step_ops = step.get_ops(workpiece)
                x, y = workpiece.pos
                ymax = config.machine.dimensions[1]
                translate_y = ymax - y - workpiece.size[1]
                step_ops.translate(x, translate_y)
                if optimize:
                    Optimize().run(step_ops)
                ops += step_ops*step.passes
        return ops
