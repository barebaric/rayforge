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
        self._importer_ref_for_pyreverse: Importer

        # The matrix is the single source of truth. A new workpiece starts
        # as a 1x1mm object at the origin.
        self._matrix: np.ndarray = np.identity(4)
        self._rebuild_matrix((0.0, 0.0), 0.0, (1.0, 1.0))

    @property
    def matrix(self) -> np.ndarray:
        """
        The 4x4 transformation matrix that defines the workpiece's position,
        rotation, and scale in world space.
        """
        return self._matrix

    @matrix.setter
    def matrix(self, value: np.ndarray):
        """
        Sets the transformation matrix. This is a transform-only operation
        that fires the `transform_changed` signal.
        """
        if np.allclose(self._matrix, value):
            return
        self._matrix = value
        self.transform_changed.send(self)

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
        size: Tuple[float, float],
    ):
        """
        Constructs the world transformation matrix from high-level properties.
        This is the single authority for creating the matrix. It transforms a
        1x1 unit object (defined on [0,1]x[0,1]) into its final world-space
        size, rotation, and position.

        The final matrix is a composition of:
        M = T_pos @ T_rot_center @ R @ T_inv_rot_center @ S
        Where:
        - S: Scales the unit object to the target `size`.
        - T_inv_rot_center: Translates the scaled object's center to the
            origin.
        - R: Rotates the object around the origin.
        - T_rot_center: Translates the object back.
        - T_pos: Translates the object to its final world `pos`.
        """
        w, h = size
        angle_rad = math.radians(-angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        cx, cy = w / 2, h / 2

        # Final translation to move the object's top-left corner
        T_pos = np.array(
            [[1, 0, 0, pos[0]], [0, 1, 0, pos[1]], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # Scale
        S = np.diag([w, h, 1, 1])

        # Rotation around the object's center (cx, cy)
        T_to_origin = np.array(
            [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        R = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        T_from_origin = np.array(
            [[1, 0, 0, cx], [0, 1, 0, cy], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        M_rot_center = T_from_origin @ R @ T_to_origin

        self._matrix = T_pos @ M_rot_center @ S

    @property
    def size(self) -> Tuple[float, float]:
        """
        The world-space size (width, height) in mm, decomposed from the
        matrix's scaling components.
        """
        scale_x = np.linalg.norm(self._matrix[:3, 0])
        scale_y = np.linalg.norm(self._matrix[:3, 1])
        return float(scale_x), float(scale_y)

    def set_size(self, width_mm: float, height_mm: float):
        """
        Sets the workpiece size in mm while preserving its world-space center
        point. This is a data-changing operation that rebuilds the
        transformation matrix and fires the `changed` signal.
        """
        new_size = float(width_mm), float(height_mm)
        if new_size == self.size:
            return

        # 1. Get current world-space center of the unit object
        old_center_world = self._matrix @ np.array([0.5, 0.5, 0, 1])

        # 2. Rebuild a temporary matrix with the new size at the origin to find
        #    where its center would land.
        temp_matrix = np.copy(self._matrix)
        self._rebuild_matrix((0, 0), self.angle, new_size)
        new_center_at_origin = self._matrix @ np.array([0.5, 0.5, 0, 1])
        self._matrix = temp_matrix  # Restore original matrix for now

        # 3. Calculate the required top-left `pos` to move the new center to
        #    the old center's location.
        new_pos_x = old_center_world[0] - new_center_at_origin[0]
        new_pos_y = old_center_world[1] - new_center_at_origin[1]

        # 4. Rebuild the final matrix with the correct new size and position.
        self._rebuild_matrix((new_pos_x, new_pos_y), self.angle, new_size)
        self.changed.send(self)

    @property
    def pos(self) -> Tuple[float, float]:
        """
        The position (in mm) of the workpiece's top-left corner, as if it
        were un-rotated. This is decomposed from the transformation matrix.
        """
        # Decompose size and angle first, as they are needed to reconstruct
        # the rotation-and-scale-only part of the matrix (M_sr).
        w, h = self.size
        angle_deg = self.angle
        angle_rad = math.radians(-angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)

        # Reconstruct the translation component of the M_sr matrix. This
        # represents the displacement caused by rotating around the center.
        cx, cy = w / 2, h / 2
        # This is the (I - R) @ center calculation
        msr_tx = cx - c * cx + s * cy
        msr_ty = cy - s * cx - c * cy

        # The final matrix is M = T_pos @ M_sr. The translation part of M is
        # T_pos + (translation part of M_sr).
        # So, T_pos = M_trans - M_sr_trans.
        return self._matrix[0, 3] - msr_tx, self._matrix[1, 3] - msr_ty

    @pos.setter
    def pos(self, new_pos: Tuple[float, float]):
        """
        Sets the position and rebuilds the transformation matrix. This is a
        transform-only operation and fires the `transform_changed` signal.
        """
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
        scale_x = np.linalg.norm(self._matrix[:3, 0])
        if scale_x == 0:
            return 0.0
        # Reconstruct the rotation matrix part by un-scaling the columns
        unscaled_x_axis = self._matrix[:2, 0] / scale_x
        angle_rad = math.atan2(unscaled_x_axis[1], unscaled_x_axis[0])
        return -math.degrees(angle_rad) % 360

    @angle.setter
    def angle(self, new_angle: float):
        """
        Sets the angle and rebuilds the transformation matrix. This is a
        transform-only operation and fires the `transform_changed` signal.
        """
        new_angle_float = float(new_angle % 360)
        if math.isclose(new_angle_float, self.angle, abs_tol=1e-9):
            return
        self._rebuild_matrix(self.pos, new_angle_float, self.size)
        self.transform_changed.send(self)

    def get_world_transform(self) -> "np.ndarray":
        """
        Returns the transformation matrix for this workpiece. The matrix is
        the single source of truth for position, size, and rotation.
        """
        return self._matrix

    def get_all_workpieces(self) -> List["WorkPiece"]:
        """For a single WorkPiece, this just returns itself in a list."""
        return [self]

    def __getstate__(self) -> Dict[str, Any]:
        """
        Prepares the object's state for pickling.
        """
        state = self.__dict__.copy()
        state.pop("_parent", None)
        state.pop("importer", None)
        state.pop("_importer_ref_for_pyreverse", None)
        state.pop("changed", None)
        state.pop("transform_changed", None)
        rclass = self.importer_class
        state["_importer_class_name"] = rclass.__name__
        state.pop("importer_class", None)
        state["matrix"] = self._matrix.tolist()
        state.pop("_matrix", None)
        return state

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restores the object's state from the pickled state.
        """
        importer_class_name = state.pop("_importer_class_name")
        self.importer_class = importer_by_name[importer_class_name]
        self._matrix = np.array(state.pop("matrix"))
        self.__dict__.update(state)
        self.importer = self.importer_class(self._data)
        self.changed = Signal()
        self.transform_changed = Signal()
        self._parent = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a pickleable dictionary. The matrix
        is the only geometric property that needs to be saved.
        """
        rclass = self.importer_class
        return {
            "uid": self.uid,
            "name": self.source_file,
            "matrix": self._matrix.tolist(),
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
            wp._matrix = np.array(data_dict["matrix"])
        # Backward compatibility for old format with 'size'
        elif "size" in data_dict and data_dict["size"] is not None:
            pos = data_dict.get("pos", (0.0, 0.0))
            angle = data_dict.get("angle", 0.0)
            size = data_dict["size"]
            wp._rebuild_matrix(pos, angle, size)

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
