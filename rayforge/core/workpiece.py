import logging
import uuid
import cairo
import importlib
from typing import (
    Generator,
    Optional,
    Tuple,
    cast,
    Dict,
    Any,
    Type,
    TYPE_CHECKING,
)
from blinker import Signal
from pathlib import Path
from ..importer import Renderer
if TYPE_CHECKING:
    from .doc import Doc
    from .layer import Layer


logger = logging.getLogger(__name__)


class WorkPiece:
    """
    Represents a real-world workpiece.

    It holds the raw source data (e.g., for an SVG or image) and manages
    a live renderer instance for operations. It also stores its position
    and size on the canvas.
    """

    def __init__(self, name: str, data: bytes, renderer_class: Type[Renderer]):
        self.layer: Optional['Layer'] = None
        self.name = name
        self.uid = str(uuid.uuid4())
        self._data = data
        self.renderer_class = renderer_class

        # The renderer is a live instance created from the raw data.
        self.renderer: Renderer = self.renderer_class(self._data)
        self._renderer_ref_for_pyreverse: Renderer

        self.pos: Optional[Tuple[float, float]] = None  # in mm
        self.size: Optional[Tuple[float, float]] = None  # in mm
        self.angle: float = 0.0  # in degrees
        self.changed: Signal = Signal()
        self.pos_changed: Signal = Signal()
        self.size_changed: Signal = Signal()
        self.angle_changed: Signal = Signal()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepares the object's state for pickling.

        This method removes live, unpickleable objects like the renderer
        instance and blinker signals. It also converts the renderer class
        type into a serializable string path for reconstruction.
        """
        state = self.__dict__.copy()

        # Remove live objects that cannot or should not be pickled.
        state.pop("layer", None)
        state.pop("renderer", None)
        state.pop("_renderer_ref_for_pyreverse", None)
        state.pop("changed", None)
        state.pop("pos_changed", None)
        state.pop("size_changed", None)
        state.pop("angle_changed", None)

        # Convert the renderer class type to a serializable string path.
        # The type object itself can be tricky to pickle directly.
        rclass = self.renderer_class
        state["_renderer_class_path"] = (
            f"{rclass.__module__}.{rclass.__name__}"
        )
        state.pop("renderer_class", None)

        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restores the object's state from the pickled state.

        This method re-imports the renderer class, re-creates the live
        renderer instance, and re-initializes the blinker signals.
        """
        # Restore the renderer class from its stored path.
        renderer_class_path = state.pop("_renderer_class_path")
        module_path, class_name = renderer_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        self.renderer_class = getattr(module, class_name)

        # Restore the rest of the pickled attributes.
        self.__dict__.update(state)

        # Re-create the live objects that were not included in the pickled
        # state.
        self.renderer = self.renderer_class(self._data)
        self.changed = Signal()
        self.pos_changed = Signal()
        self.size_changed = Signal()
        self.angle_changed = Signal()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a pickleable dictionary.
        The live renderer instance is not serialized; instead, the raw
        data and renderer class path are stored for reconstruction.
        """
        rclass = self.renderer_class
        return {
            "uid": self.uid,
            "name": self.name,
            "pos": self.pos,
            "size": self.size,
            "angle": self.angle,
            "data": self._data,
            "renderer": f"{rclass.__module__}.{rclass.__name__}",
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "WorkPiece":
        """
        Deserializes a WorkPiece from a dictionary by reconstructing it
        from its raw data and renderer class.
        """
        # Dynamically import the renderer class from its path
        module_path, class_name = data_dict['renderer'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        renderer_class = getattr(module, class_name)

        # Create the WorkPiece instance using the main constructor
        wp = cls(data_dict['name'], data_dict['data'], renderer_class)

        # Restore state
        wp.uid = data_dict.get('uid', uuid.uuid4())
        wp.pos = data_dict.get('pos')
        wp.size = data_dict.get('size')
        wp.angle = data_dict.get('angle', 0.0)

        return wp

    @property
    def doc(self) -> Optional["Doc"]:
        if not self.layer:
            return None
        return self.layer.doc

    def set_pos(self, x_mm: float, y_mm: float):
        new_pos = float(x_mm), float(y_mm)
        if new_pos == self.pos:
            return
        self.pos = new_pos
        self.changed.send(self)
        self.pos_changed.send(self)

    def set_size(self, width_mm: float, height_mm: float):
        new_size = float(width_mm), float(height_mm)
        if new_size == self.size:
            return
        self.size = new_size
        self.changed.send(self)
        self.size_changed.send(self)

    def set_angle(self, angle: float):
        new_angle = float(angle % 360)
        if new_angle == self.angle:
            return
        self.angle = new_angle
        self.changed.send(self)
        self.angle_changed.send(self)

    def get_default_size(
        self, bounds_width: float, bounds_height: float
    ) -> Tuple[float, float]:
        """Calculates a sensible default size based on the content's aspect
        ratio and the provided container bounds."""
        size = self.renderer.get_natural_size()
        if None not in size:
            return cast(Tuple[float, float], size)

        aspect = self.get_default_aspect_ratio()
        width_mm = bounds_width
        height_mm = width_mm / aspect if aspect else bounds_height
        if height_mm > bounds_height:
            height_mm = bounds_height
            width_mm = height_mm * aspect if aspect else bounds_width

        return width_mm, height_mm

    def get_current_size(self) -> Optional[Tuple[float, float]]:
        """Returns the currently set size (in mm), or None if not set."""
        return self.size

    def get_default_aspect_ratio(self):
        return self.renderer.get_aspect_ratio()

    def get_current_aspect_ratio(self) -> Optional[float]:
        return (self.size[0] / self.size[1]
                if self.size and self.size[1] else None)

    @classmethod
    def from_file(cls, filename: Path, renderer_class: type[Renderer]):
        data = filename.read_bytes()
        wp = cls(filename.name, data, renderer_class)
        return wp

    def render_for_ops(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
        size: Optional[Tuple[float, float]] = None
    ) -> Optional[cairo.ImageSurface]:
        """Renders to a pixel surface at the workpiece's current size, or a
        provided override size. Returns None if no size is available."""
        current_size = self.get_current_size() if size is None else size
        if not current_size:
            return None

        width_mm, height_mm = current_size

        target_width_px = int(width_mm * pixels_per_mm_x)
        target_height_px = int(height_mm * pixels_per_mm_y)

        return self.renderer.render_to_pixels(
            width=target_width_px, height=target_height_px
        )

    def render_chunk(
        self,
        pixels_per_mm_x: int,
        pixels_per_mm_y: int,
        size: Optional[Tuple[float, float]] = None,
        max_chunk_width: Optional[int] = None,
        max_chunk_height: Optional[int] = None,
        max_memory_size: Optional[int] = None,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        """Renders in chunks at the workpiece's current size, or a provided
        override size. Yields nothing if no size is available."""
        current_size = self.get_current_size() if size is None else size
        if not current_size:
            return

        width = int(current_size[0] * pixels_per_mm_x)
        height = int(current_size[1] * pixels_per_mm_y)

        for chunk in self.renderer.render_chunk(
            width,
            height,
            max_chunk_width=max_chunk_width,
            max_chunk_height=max_chunk_height,
            max_memory_size=max_memory_size,
        ):
            yield chunk

    def dump(self, indent=0):
        print("  " * indent, self.name, self.renderer.label)

    @property
    def pos_machine(self) -> Optional[Tuple[float, float]]:
        """
        Gets the workpiece's anchor position in the machine's native
        coordinate system.
        """
        if self.pos is None or self.size is None:
            return None

        from ..config import config

        if config.machine is None:
            return None

        model_x, model_y = self.pos  # Canonical: Y-up, bottom-left

        if config.machine.y_axis_down:
            # Convert to machine: Y-down, top-left
            machine_height = config.machine.dimensions[1]
            model_h = self.size[1]
            machine_y = machine_height - model_y - model_h
            return model_x, machine_y
        else:
            # Machine is Y-up, same as model
            return self.pos

    @pos_machine.setter
    def pos_machine(self, pos: Tuple[float, float]):
        """
        Sets the workpiece's position from the machine's native
        coordinate system.
        """
        if pos is None or self.size is None:
            return

        from ..config import config

        if config.machine is None:
            return None

        machine_x, machine_y = pos
        model_pos = (0.0, 0.0)

        if config.machine.y_axis_down:
            # Convert from machine (Y-down, top-left) to
            # model (Y-up, bottom-left)
            machine_height = config.machine.dimensions[1]
            model_h = self.size[1]
            model_y = machine_height - machine_y - model_h
            model_pos = machine_x, model_y
        else:
            # Machine is Y-up, same as model
            model_pos = machine_x, machine_y

        self.set_pos(model_pos[0], model_pos[1])
