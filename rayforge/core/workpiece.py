import logging
import uuid
import math
import cairo
import numpy as np
from typing import (
    Generator,
    Optional,
    Tuple,
    cast,
    Dict,
    Any,
    Type,
    TYPE_CHECKING,
    List,
)
from blinker import Signal
from pathlib import Path
from ..importer import Importer, importer_by_name
from .item import DocItem

if TYPE_CHECKING:
    from .layer import Layer


logger = logging.getLogger(__name__)


class WorkPiece(DocItem):
    """
    Represents a real-world workpiece.

    It holds the raw source data (e.g., for an SVG or image) and manages
    a live importer instance for operations. Its position, rotation, and size
    are managed through a single transformation matrix, which serves as the
    single source of truth for its placement on the canvas.
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
        self._importer_ref_for_pyreverse: Importer

        # Geometric properties. The matrix is the source of truth for position
        # and angle. Size is stored directly as it's needed for pre-transform
        # rendering and ops generation.
        self.size: Optional[Tuple[float, float]] = None  # in mm
        self.matrix: np.ndarray = np.identity(4)
        self._rebuild_matrix((0.0, 0.0), 0.0, None)

    @property
    def layer(self) -> Optional["Layer"]:
        return self.parent

    @layer.setter
    def layer(self, value: Optional["Layer"]):
        self.parent = value

    def _rebuild_matrix(
        self,
        pos: Tuple[float, float],
        angle_deg: float,
        size: Optional[Tuple[float, float]],
    ):
        """
        Constructs the world transformation matrix from pos, angle, and size,
        and sets it as the instance's authoritative matrix.
        """
        # A positive angle in the UI/model corresponds to a clockwise
        # rotation. Standard math libs use positive for counter-clockwise,
        # so we must negate the angle here to match the visual convention.
        angle_rad = math.radians(-angle_deg)

        # Start with a pure rotation matrix around the origin (0,0).
        c = math.cos(angle_rad)
        s = math.sin(angle_rad)
        matrix = np.array(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        # When an object with size is rotated "in-place" around its center,
        # its local origin (0,0) is displaced. We need to calculate this
        # displacement and add it to the final world translation.
        if size:
            cx, cy = size[0] / 2, size[1] / 2

            # The vector from the local origin to the center.
            center_vec = np.array([cx, cy, 0, 1])

            # Find where the center point would land if it were rotated
            # around the origin.
            rotated_center_vec = matrix @ center_vec

            # The displacement required to move the object back so that its
            # center aligns with its original un-rotated position is the
            # difference between the original center and the rotated center.
            dx = cx - rotated_center_vec[0]
            dy = cy - rotated_center_vec[1]

            # The final translation is the object's world position plus this
            # calculated rotational displacement.
            final_tx = pos[0] + dx
            final_ty = pos[1] + dy
            matrix[0, 3] = final_tx
            matrix[1, 3] = final_ty
        else:
            # If there is no size, the center is the origin, so there is no
            # rotational displacement. The translation is just the world
            # position.
            matrix[0, 3] = pos[0]
            matrix[1, 3] = pos[1]

        self.matrix = matrix

    @property
    def pos(self) -> Tuple[float, float]:
        """
        The position (in mm) of the workpiece's top-left corner, as if it
        were un-rotated. This is decomposed from the transformation matrix.
        """
        angle_rad = math.atan2(self.matrix[1, 0], self.matrix[0, 0])
        rot_matrix = np.array(
            [
                [math.cos(angle_rad), -math.sin(angle_rad), 0, 0],
                [math.sin(angle_rad), math.cos(angle_rad), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=float,
        )

        dx, dy = 0.0, 0.0
        if self.size:
            cx, cy = self.size[0] / 2, self.size[1] / 2
            center_vec = np.array([cx, cy, 0, 1])
            rotated_center_vec = rot_matrix @ center_vec
            dx = cx - rotated_center_vec[0]
            dy = cy - rotated_center_vec[1]

        final_tx = self.matrix[0, 3]
        final_ty = self.matrix[1, 3]

        return final_tx - dx, final_ty - dy

    @pos.setter
    def pos(self, new_pos: Tuple[float, float]):
        """Sets the position and rebuilds the transformation matrix."""
        new_pos_float = float(new_pos[0]), float(new_pos[1])
        if new_pos_float == self.pos:
            return
        self._rebuild_matrix(new_pos_float, self.angle, self.size)
        self.transform_changed.send(self)

    @property
    def angle(self) -> float:
        """
        The clockwise rotation angle (in degrees) of the workpiece.
        This is decomposed from the transformation matrix.
        """
        # The matrix stores rotation for a negative angle, so we negate the
        # result to get the user-facing positive angle.
        angle_rad = math.atan2(self.matrix[1, 0], self.matrix[0, 0])
        return -math.degrees(angle_rad) % 360

    @angle.setter
    def angle(self, new_angle: float):
        """Sets the angle and rebuilds the transformation matrix."""
        new_angle_float = float(new_angle % 360)
        if new_angle_float == self.angle:
            return
        self._rebuild_matrix(self.pos, new_angle_float, self.size)
        self.transform_changed.send(self)

    def get_world_transform(self) -> "np.ndarray":
        """
        Returns the transformation matrix for this workpiece. The matrix is
        the single source of truth for position and rotation.
        """
        return self.matrix

    def get_all_workpieces(self) -> List["WorkPiece"]:
        """For a single WorkPiece, this just returns itself in a list."""
        return [self]

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepares the object's state for pickling.

        This method removes live, unpickleable objects like the importer
        instance and blinker signals. It also converts the importer class
        type into a serializable string name for reconstruction.
        """
        state = self.__dict__.copy()

        # Remove live objects that cannot or should not be pickled.
        state.pop("_parent", None)
        state.pop("importer", None)
        state.pop("_importer_ref_for_pyreverse", None)
        state.pop("changed", None)
        state.pop("transform_changed", None)

        # Convert the importer class type to a serializable string name.
        # The class object itself can be tricky to pickle directly.
        rclass = self.importer_class
        state["_importer_class_name"] = rclass.__name__
        state.pop("importer_class", None)

        # Convert numpy matrix to a list for serialization
        state["matrix"] = state["matrix"].tolist()

        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restores the object's state from the pickled state.

        This method re-creates the importer class from its name,
        re-creates the live importer instance, and re-initializes the
        blinker signals.
        """
        # Restore the importer class from its stored name using the registry.
        importer_class_name = state.pop("_importer_class_name")
        self.importer_class = importer_by_name[importer_class_name]

        # Restore the rest of the pickled attributes.
        self.__dict__.update(state)

        # Convert list back to numpy matrix
        self.matrix = np.array(self.matrix)

        # Re-create the live objects that were not included in the pickled
        # state.
        self.importer = self.importer_class(self._data)
        self.changed = Signal()
        self.transform_changed = Signal()
        self._parent = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a pickleable dictionary.
        The live importer instance is not serialized; instead, the raw
        data and importer class name are stored for reconstruction.
        """
        rclass = self.importer_class
        return {
            "uid": self.uid,
            "name": self.source_file,
            "size": self.size,
            "matrix": self.matrix.tolist(),
            "data": self._data,
            "importer": rclass.__name__,
        }

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Any]) -> "WorkPiece":
        """
        Deserializes a WorkPiece from a dictionary by reconstructing it
        from its raw data and importer class.
        """
        # Look up the importer class from its name via the central registry.
        importer_class = importer_by_name[data_dict["importer"]]

        # Create the WorkPiece instance using the main constructor
        wp = cls(data_dict["name"], data_dict["data"], importer_class)

        # Restore state
        wp.uid = data_dict.get("uid", str(uuid.uuid4()))
        wp.name = data_dict.get("name") or wp.source_file.name
        wp.size = data_dict.get("size")

        # Restore matrix, with backward compatibility for old format
        if "matrix" in data_dict:
            wp.matrix = np.array(data_dict["matrix"])
        else:
            # Default state if no transform info is present
            wp._rebuild_matrix((0.0, 0.0), 0.0, wp.size)

        return wp

    def set_pos(self, x_mm: float, y_mm: float):
        """Legacy method, use property `pos` instead."""
        self.pos = (x_mm, y_mm)

    def set_size(self, width_mm: float, height_mm: float):
        new_size = float(width_mm), float(height_mm)
        if new_size == self.size:
            return
        # Get pos/angle before changing size, as they depend on it
        current_pos = self.pos
        current_angle = self.angle
        self.size = new_size
        # Rebuild matrix with the new size
        self._rebuild_matrix(current_pos, current_angle, self.size)
        self.changed.send(self)

    def set_angle(self, angle: float):
        """Legacy method, use property `angle` instead."""
        self.angle = angle

    def get_default_size(
        self, bounds_width: float, bounds_height: float
    ) -> Tuple[float, float]:
        """Calculates a sensible default size based on the content's aspect
        ratio and the provided container bounds."""
        size = self.importer.get_natural_size()
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
        return self.importer.get_aspect_ratio()

    def get_current_aspect_ratio(self) -> Optional[float]:
        return (
            self.size[0] / self.size[1] if self.size and self.size[1] else None
        )

    @classmethod
    def from_file(cls, filename: Path, importer_class: type[Importer]):
        data = filename.read_bytes()
        wp = cls(filename, data, importer_class)
        return wp

    def render_for_ops(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
        size: Optional[Tuple[float, float]] = None,
    ) -> Optional[cairo.ImageSurface]:
        """Renders to a pixel surface at the workpiece's current size, or a
        provided override size. Returns None if no size is available."""
        current_size = self.get_current_size() if size is None else size
        if not current_size:
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

        for chunk in self.importer.render_chunk(
            width,
            height,
            max_chunk_width=max_chunk_width,
            max_chunk_height=max_chunk_height,
            max_memory_size=max_memory_size,
        ):
            yield chunk

    def dump(self, indent=0):
        print("  " * indent, self.source_file, self.importer.label)

    @property
    def pos_machine(self) -> Optional[Tuple[float, float]]:
        """
        Gets the workpiece's anchor position in the machine's native
        coordinate system.
        """
        current_pos = self.pos
        if current_pos is None or self.size is None:
            return None

        from ..config import config

        if config.machine is None:
            return None

        model_x, model_y = current_pos  # Canonical: Y-up, bottom-left

        if config.machine.y_axis_down:
            # Convert to machine: Y-down, top-left
            machine_height = config.machine.dimensions[1]
            model_h = self.size[1]
            machine_y = machine_height - model_y - model_h
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

        self.pos = model_pos
