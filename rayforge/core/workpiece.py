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
    List,
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
        self._importer_ref_for_pyreverse: Importer

        # The matrix is the single source of truth. A new workpiece starts
        # as a 1x1mm object at the origin. _matrix is inherited from DocItem.
        self._rebuild_matrix((0.0, 0.0), 0.0, (1.0, 1.0))

    @property
    def layer(self) -> Optional["Layer"]:
        return self.parent

    @layer.setter
    def layer(self, value: Optional["Layer"]):
        self.parent = value

    @property
    def matrix(self) -> "Matrix":
        """The 3x3 local transformation matrix for this item."""
        return self._matrix

    @matrix.setter
    def matrix(self, value: "Matrix"):
        """
        Sets the local transformation matrix and fires the appropriate signals.

        This setter intelligently determines if the size (scale) of the
        workpiece has changed.
        - If scale changes, it fires `changed` (for ops regeneration) and
          `transform_changed` (for UI updates).
        - If only position/rotation changes, it only fires `transform_changed`.
        """
        old_matrix = self._matrix
        if old_matrix == value:
            return

        old_scale = old_matrix.get_abs_scale()
        new_scale = value.get_abs_scale()

        self._matrix = value

        # Use a tolerance for floating point comparison of scale.
        scale_changed = not (
            abs(old_scale[0] - new_scale[0]) < 1e-9
            and abs(old_scale[1] - new_scale[1]) < 1e-9
        )

        # If the size/scale changed, it's a data-level change.
        if scale_changed:
            self.changed.send(self)

        # Any geometric change requires a transform update for the UI.
        self.transform_changed.send(self)

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
        M = T_pos @ R_center @ S
        Where:
        - S: Scales the unit object to the target `size`.
        - R_center: Rotates the scaled object around its center.
        - T_pos: Translates the object to its final world `pos`.
        """
        w, h = size
        cx, cy = w / 2, h / 2

        # Use Matrix class methods to build transformations
        S = Matrix.scale(w, h)
        # The angle is a standard counter-clockwise rotation.
        R = Matrix.rotation(angle_deg, center=(cx, cy))
        T = Matrix.translation(pos[0], pos[1])

        # Compose the final matrix. Order is critical: scale first,
        # then rotate around center, then translate to final position.
        self.matrix = T @ R @ S

    @property
    def size(self) -> Tuple[float, float]:
        """
        The world-space size (width, height) in mm, as absolute values,
        decomposed from the matrix's scaling components.
        """
        return self.matrix.get_abs_scale()

    def set_size(self, width_mm: float, height_mm: float):
        """
        Sets the workpiece size in mm while preserving its world-space center
        point. This is a data-changing operation that rebuilds the
        transformation matrix and fires the `changed` signal.
        """
        new_size = float(width_mm), float(height_mm)
        current_w, current_h = self.size
        if (
            abs(new_size[0] - current_w) < 1e-9
            and abs(new_size[1] - current_h) < 1e-9
        ):
            return

        # 1. Get current world-space center of the unit object
        old_center_world = self.matrix.transform_point((0.5, 0.5))

        # 2. Build a temporary matrix with the new size and current angle
        #    at the origin to find its center point. This avoids mutating the
        #    instance and firing signals prematurely.
        w, h = new_size
        cx, cy = w / 2, h / 2
        S_temp = Matrix.scale(w, h)
        R_temp = Matrix.rotation(self.angle, center=(cx, cy))
        # The temporary matrix is composed of scale then rotate around center
        temp_matrix_at_origin = R_temp @ S_temp
        new_center_at_origin = temp_matrix_at_origin.transform_point(
            (0.5, 0.5)
        )

        # 3. Calculate the required top-left `pos` to move the new center to
        #    the old center's location.
        new_pos_x = old_center_world[0] - new_center_at_origin[0]
        new_pos_y = old_center_world[1] - new_center_at_origin[1]

        # 4. Rebuild the final matrix with the correct new size and position.
        # This will call the matrix.setter, which will fire the correct
        # signals.
        self._rebuild_matrix((new_pos_x, new_pos_y), self.angle, new_size)

    @property
    def pos(self) -> Tuple[float, float]:
        """
        The position (in mm) of the workpiece's top-left corner, as if it
        were un-rotated. This is decomposed from the transformation matrix.
        """
        w, h = self.size
        angle = self.angle

        S = Matrix.scale(w, h)
        R = Matrix.rotation(angle, center=(w / 2, h / 2))
        # This matrix represents the combined scale and centered rotation
        M_rs = R @ S

        # The final matrix is M = T @ M_rs.
        # The translation of M is T_trans + M_rs_trans.
        # So, T_trans = M_trans - M_rs_trans.
        m_tx, m_ty = self.matrix.get_translation()
        mrs_tx, mrs_ty = M_rs.get_translation()

        return (m_tx - mrs_tx, m_ty - mrs_ty)

    @pos.setter
    def pos(self, new_pos: Tuple[float, float]):
        """
        Sets the position and rebuilds the transformation matrix. This is a
        transform-only operation and fires the `transform_changed` signal.
        """
        new_pos_float = float(new_pos[0]), float(new_pos[1])
        if new_pos_float == self.pos:
            return
        # This will call the matrix.setter, which will fire the correct
        # signals.
        self._rebuild_matrix(new_pos_float, self.angle, self.size)

    @property
    def angle(self) -> float:
        """
        The rotation angle (in degrees) of the workpiece.
        This is decomposed from the transformation matrix.
        """
        return self.matrix.get_rotation() % 360

    @angle.setter
    def angle(self, new_angle: float):
        """
        Sets the angle and rebuilds the transformation matrix. This is a
        transform-only operation and fires the `transform_changed` signal.
        """
        new_angle_float = float(new_angle % 360)
        # Use a small tolerance for floating point comparison of angles
        current_angle = self.angle
        if (
            abs(new_angle_float - current_angle) < 1e-9
            or abs(new_angle_float - current_angle - 360) < 1e-9
        ):
            return
        # This will call the matrix.setter, which will fire the correct
        # signals.
        self._rebuild_matrix(self.pos, new_angle_float, self.size)

    def get_world_transform(self) -> "Matrix":
        """
        Returns the transformation matrix for this workpiece. The matrix is
        the single source of truth for position, size, and rotation.
        """
        # The parent (Layer) is not a DocItem, so world transform is just the
        # local matrix. This correctly overrides DocItem's implementation.
        return self.matrix

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
