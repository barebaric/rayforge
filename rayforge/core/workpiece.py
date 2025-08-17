import logging
import uuid
import cairo
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
from ..importer import Importer, importer_by_name
from .item import DocItem
from .matrix import Matrix

if TYPE_CHECKING:
    from .layer import Layer


logger = logging.getLogger(__name__)


class WorkPiece(DocItem):
    """
    Represents a real-world workpiece.

    It holds the raw source data (e.g., for an SVG or image) and manages
    a live importer instance for operations. Its position, scale/size, and
    rotation are managed through a single transformation matrix, which serves
    as the single source of truth for its placement on the canvas.
    """

    def __init__(
        self,
        source_file: Path,
        data: bytes,
        importer_class: Type[Importer],
    ):
        super().__init__(name=source_file.name)
        self.source_file = source_file
        self._data = data
        self.importer_class = importer_class

        # The importer is a live instance created from the raw data.
        self.importer: Importer = self.importer_class(self._data)

    @property
    def layer(self) -> Optional["Layer"]:
        """Traverses the hierarchy to find the parent Layer."""
        from .layer import Layer  # Local import to avoid circular dependency

        p = self.parent
        while p:
            if isinstance(p, Layer):
                return p
            p = p.parent
        return None

    def in_world(self) -> "WorkPiece":
        """
        Returns a new, unparented WorkPiece instance whose local
        transformation matrix is the world transformation matrix of this one.
        This effectively "bakes" the parent transformations into the object,
        making it suitable for serialization or use in contexts without a
        document hierarchy.
        """
        # Create a new instance to avoid side effects with signals,
        # parents, etc.
        world_wp = WorkPiece(self.source_file, self._data, self.importer_class)
        world_wp.uid = self.uid  # Preserve UID for tracking
        world_wp.name = self.name
        world_wp.matrix = self.get_world_transform()
        return world_wp

    def get_local_size(self) -> Tuple[float, float]:
        """
        The local-space size (width, height) in mm, as absolute values,
        decomposed from the local transformation matrix. This is used for
        determining rasterization resolution.
        """
        return self.matrix.get_abs_scale()

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepares the object's state for pickling.
        """
        state = self.__dict__.copy()
        state.pop("_parent", None)
        state.pop("children", None)
        state.pop("importer", None)

        # Pop all signals defined in DocItem as they cannot be pickled
        state.pop("updated", None)
        state.pop("transform_changed", None)
        state.pop("descendant_added", None)
        state.pop("descendant_removed", None)
        state.pop("descendant_updated", None)
        state.pop("descendant_transform_changed", None)

        rclass = self.importer_class
        state["_importer_class_name"] = rclass.__name__
        state.pop("importer_class", None)
        state["matrix"] = self._matrix.m.tolist()
        state.pop("_matrix", None)
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restores the object's state from the pickled state.
        """
        importer_class_name = state.pop("_importer_class_name")
        self.importer_class = importer_by_name[importer_class_name]
        self._matrix = Matrix(state.pop("matrix"))
        self.__dict__.update(state)
        self.importer = self.importer_class(self._data)

        # Re-initialize signals as they are not pickled.
        self.updated = Signal()
        self.transform_changed = Signal()
        self.descendant_added = Signal()
        self.descendant_removed = Signal()
        self.descendant_updated = Signal()
        self.descendant_transform_changed = Signal()

        self._parent = None
        self.children = []

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a pickleable dictionary. The matrix
        is the only geometric property that needs to be saved.
        """
        rclass = self.importer_class
        return {
            "uid": self.uid,
            "name": self.source_file,
            "matrix": self._matrix.m.tolist(),
            "data": self._data,
            "importer": rclass.__name__,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "WorkPiece":
        """
        Deserializes a WorkPiece from a dictionary.
        """
        importer_class = importer_by_name[data_dict["importer"]]
        wp = cls(data_dict["name"], data_dict["data"], importer_class)
        wp.uid = data_dict.get("uid", str(uuid.uuid4()))
        wp.name = data_dict.get("name") or wp.source_file.name

        if "matrix" in data_dict:
            wp.matrix = Matrix(data_dict["matrix"])
        # Backward compatibility for old format with 'size'
        elif "size" in data_dict and data_dict["size"] is not None:
            # Reconstruct matrix from old properties
            pos = data_dict.get("pos", (0.0, 0.0))
            angle = data_dict.get("angle", 0.0)
            size = data_dict["size"]
            w, h = size
            cx, cy = w / 2, h / 2
            S = Matrix.scale(w, h)
            R = Matrix.rotation(angle, center=(cx, cy))
            T = Matrix.translation(pos[0], pos[1])
            wp.matrix = T @ R @ S

        return wp

    def set_pos(self, x_mm: float, y_mm: float):
        """Legacy method, use property `pos` instead."""
        self.pos = (x_mm, y_mm)

    def set_angle(self, angle: float):
        """Legacy method, use property `angle` instead."""
        self.angle = angle

    def get_default_size(
        self, bounds_width: float, bounds_height: float
    ) -> Tuple[float, float]:
        """Calculates a sensible default size based on the content's aspect
        ratio and the provided container bounds."""
        size = self.importer.get_natural_size()
        if size and None not in size:
            return cast(Tuple[float, float], size)

        aspect = self.get_default_aspect_ratio()
        if aspect is None:
            return bounds_width, bounds_height

        width_mm = bounds_width
        height_mm = width_mm / aspect
        if height_mm > bounds_height:
            height_mm = bounds_height
            width_mm = height_mm * aspect

        return width_mm, height_mm

    def get_default_aspect_ratio(self):
        return self.importer.get_aspect_ratio()

    def get_current_aspect_ratio(self) -> Optional[float]:
        w, h = self.size
        return w / h if h else None

    @classmethod
    def from_file(cls, filename: Path, importer_class: type[Importer]):
        data = filename.read_bytes()
        wp = cls(filename, data, importer_class)

        # A new workpiece is created at 1x1mm. We must immediately resize it
        # to a sensible default based on its natural size and the machine
        # dimensions.
        from ..config import config

        bounds_w, bounds_h = (
            config.machine.dimensions if config.machine else (100.0, 100.0)
        )
        default_w, default_h = wp.get_default_size(bounds_w, bounds_h)
        wp.set_size(default_w, default_h)

        return wp

    def render_for_ops(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
    ) -> Optional[cairo.ImageSurface]:
        """Renders to a pixel surface at the workpiece's current size.
        Returns None if size is not valid."""
        # Use the final world-space size for rendering resolution. This is
        # critical for preserving quality when scaling is applied to a
        # parent group.
        current_size = self.size
        if not current_size or current_size[0] <= 0 or current_size[1] <= 0:
            return None

        width_mm, height_mm = current_size
        target_width_px = int(width_mm * pixels_per_mm_x)
        target_height_px = int(height_mm * pixels_per_mm_y)

        return self.importer.render_to_pixels(
            width=target_width_px, height=target_height_px
        )

    def render_chunk(
        self,
        pixels_per_mm_x: int,
        pixels_per_mm_y: int,
        max_chunk_width: Optional[int] = None,
        max_chunk_height: Optional[int] = None,
        max_memory_size: Optional[int] = None,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        """Renders in chunks at the workpiece's current size.
        Yields nothing if size is not valid."""
        # Use the final world-space size for rendering resolution. This is
        # critical for preserving quality when scaling is applied to a
        # parent group.
        current_size = self.size
        if not current_size or current_size[0] <= 0 or current_size[1] <= 0:
            return

        width = int(current_size[0] * pixels_per_mm_x)
        height = int(current_size[1] * pixels_per_mm_y)

        yield from self.importer.render_chunk(
            width,
            height,
            max_chunk_width=max_chunk_width,
            max_chunk_height=max_chunk_height,
            max_memory_size=max_memory_size,
        )

    def dump(self, indent=0):
        print("  " * indent, self.source_file, self.importer.label)

    @property
    def pos_machine(self) -> Optional[Tuple[float, float]]:
        """
        Gets the workpiece's anchor position in the machine's native
        coordinate system.
        """
        current_pos = self.pos
        current_size = self.size
        if current_pos is None or current_size is None:
            return None

        from ..config import config

        if config.machine is None:
            return None

        model_x, model_y = current_pos  # Canonical: Y-up, bottom-left

        if config.machine.y_axis_down:
            # Convert to machine: Y-down, top-left
            machine_height = config.machine.dimensions[1]
            machine_y = machine_height - model_y - current_size[1]
            return model_x, machine_y
        else:
            # Machine is Y-up, same as model
            return current_pos

    @pos_machine.setter
    def pos_machine(self, pos: Tuple[float, float]):
        """
        Sets the workpiece's position from the machine's native
        coordinate system.
        """
        current_size = self.size
        if pos is None or current_size is None:
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
            model_y = machine_height - machine_y - current_size[1]
            model_pos = machine_x, model_y
        else:
            # Machine is Y-up, same as model
            model_pos = machine_x, machine_y

        self.pos = model_pos
