import logging
import cairo
from typing import (
    Generator,
    Optional,
    Tuple,
    cast,
    Dict,
    Any,
    TYPE_CHECKING,
    List,
)
from pathlib import Path
import warnings
from dataclasses import asdict
from copy import deepcopy
import math
import numpy as np

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from .geo import Geometry
from .item import DocItem
from .matrix import Matrix
from .tab import Tab

if TYPE_CHECKING:
    from .layer import Layer
    from ..image.base_renderer import Renderer
    from .import_source import ImportSource


logger = logging.getLogger(__name__)


class WorkPiece(DocItem):
    """
    Represents a real-world workpiece. It is a lightweight data container,
    holding its vector geometry, its transformation matrix, and a link to its
    source. It is completely decoupled from importers.
    """

    def __init__(
        self,
        name: str,
        vectors: Optional[Geometry] = None,
    ):
        super().__init__(name=name)
        self.vectors = vectors
        """
        The normalized vector geometry defining the intrinsic shape.

        This `Geometry` object represents the workpiece's shape, normalized
        to fit within a 1x1 unit reference box. This separation of intrinsic
        shape from its world transformation is crucial for preventing
        rendering and processing errors.

        The local coordinate space of this normalized geometry has the
        following properties:

        - **Reference Size**: The geometry is scaled to fit within a box
          that is approximately 1 unit wide by 1 unit tall.
        - **Origin (0,0)**: The anchor point is the bottom-left corner of the
          geometry's bounding box.
        - **Transformation**: The vector data itself is static. All physical
          sizing, positioning, and rotation are handled by applying the
          `WorkPiece.matrix` to this normalized shape.
        """
        self.import_source_uid: Optional[str] = None

        # The cache for rendered vips images. Key is (width, height).
        # This is the proper place for this state, not monkey-patched.
        self._render_cache: Dict[Tuple[int, int], pyvips.Image] = {}

        # Transient attributes for deserialized instances in subprocesses
        self._data: Optional[bytes] = None
        self._renderer: Optional["Renderer"] = None

        self._tabs: List[Tab] = []
        self._tabs_enabled: bool = True

    def clear_render_cache(self):
        """
        Invalidates and clears all cached renders for this workpiece.
        Should be called if the underlying _data or geometry changes.
        """
        self._render_cache.clear()

    @property
    def source(self) -> "Optional[ImportSource]":
        """
        Convenience property to retrieve the full ImportSource object from the
        document's central registry.
        """
        if self.doc and self.import_source_uid:
            return self.doc.get_import_source_by_uid(self.import_source_uid)
        return None

    @property
    def data(self) -> Optional[bytes]:
        """Retrieves the raw source data."""
        # Prioritize transient data for isolated/subprocess instances
        if self._data is not None:
            return self._data
        source = self.source
        return source.data if source else None

    @property
    def source_file(self) -> Optional[Path]:
        """Retrieves the source file path from the linked ImportSource."""
        source = self.source
        return source.source_file if source else None

    @property
    def renderer(self) -> "Optional[Renderer]":
        """Retrieves the renderer."""
        # Prioritize transient renderer for isolated/subprocess instances
        if self._renderer is not None:
            return self._renderer
        source = self.source
        return source.renderer if source else None

    @property
    def tabs(self) -> List[Tab]:
        """The list of Tab objects for this workpiece."""
        return self._tabs

    @tabs.setter
    def tabs(self, new_tabs: List[Tab]):
        if self._tabs != new_tabs:
            self._tabs = new_tabs
            self.updated.send(self)

    @property
    def tabs_enabled(self) -> bool:
        return self._tabs_enabled

    @tabs_enabled.setter
    def tabs_enabled(self, new_value: bool):
        if self._tabs_enabled != new_value:
            self._tabs_enabled = new_value
            self.updated.send(self)

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
        document hierarchy. It also hydrates the instance with the necessary
        data for rendering in isolated environments like subprocesses.
        """
        # Create a new instance to avoid side effects with signals,
        # parents, etc.
        world_wp = WorkPiece(self.name, self.vectors)
        world_wp.uid = self.uid  # Preserve UID for tracking
        world_wp.matrix = self.get_world_transform()
        world_wp.tabs = deepcopy(self.tabs)
        world_wp.tabs_enabled = self.tabs_enabled
        world_wp.import_source_uid = self.import_source_uid

        # Hydrate with data and renderer for use in isolated contexts
        # like subprocesses where the document link is lost.
        source = self.source
        if source:
            # Use the public .data property to get the correct render data
            world_wp._data = source.data
            world_wp._renderer = source.renderer

        # Do NOT link back to the parent. The point of this method is to
        # create a self-contained object suitable for serialization.
        return world_wp

    def get_local_size(self) -> Tuple[float, float]:
        """
        The local-space size (width, height) in mm, as absolute values,
        decomposed from the local transformation matrix. This is used for
        determining rasterization resolution.
        """
        return self.matrix.get_abs_scale()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a dictionary.
        """
        return {
            "uid": self.uid,
            "name": self.name,
            "matrix": self._matrix.to_list(),
            "vectors": self.vectors.to_dict() if self.vectors else None,
            "tabs": [asdict(t) for t in self._tabs],
            "tabs_enabled": self._tabs_enabled,
            "import_source_uid": self.import_source_uid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPiece":
        """
        Restores a WorkPiece instance from a dictionary.
        """
        from .geo import Geometry
        from ..image import renderer_by_name

        vectors = (
            Geometry.from_dict(data["vectors"]) if data["vectors"] else None
        )

        wp = cls(
            name=data["name"],
            vectors=vectors,
        )
        wp.uid = data["uid"]
        wp.matrix = Matrix.from_list(data["matrix"])

        loaded_tabs = []
        for t_data in data.get("tabs", []):
            t_data_copy = t_data.copy()
            # Ignore 'length' for backward compatibility with older files.
            t_data_copy.pop("length", None)
            loaded_tabs.append(Tab(**t_data_copy))
        wp.tabs = loaded_tabs
        wp.tabs_enabled = data.get("tabs_enabled", True)
        wp.import_source_uid = data.get("import_source_uid")

        # Hydrate with transient data if provided for subprocesses
        if "data" in data:
            wp._data = data["data"]
        if "renderer_name" in data:
            renderer_name = data["renderer_name"]
            if renderer_name in renderer_by_name:
                wp._renderer = renderer_by_name[renderer_name]

        return wp

    def get_natural_size(self) -> Optional[Tuple[float, float]]:
        renderer = self.renderer
        return renderer.get_natural_size(self) if renderer else None

    def get_natural_aspect_ratio(self) -> Optional[float]:
        size = self.get_natural_size()
        if size:
            w, h = size
            if w and h and h > 0:
                return w / h
        return None

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
        size = self.get_natural_size()
        if size and None not in size:
            return cast(Tuple[float, float], size)

        aspect = self.get_natural_aspect_ratio()
        if aspect is None:
            return bounds_width, bounds_height

        width_mm = bounds_width
        height_mm = width_mm / aspect
        if height_mm > bounds_height:
            height_mm = bounds_height
            width_mm = height_mm * aspect

        return width_mm, height_mm

    def get_current_aspect_ratio(self) -> Optional[float]:
        w, h = self.size
        return w / h if h else None

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        renderer = self.renderer
        return (
            renderer.render_to_pixels(self, width, height)
            if renderer
            else None
        )

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

        return self.render_to_pixels(target_width_px, target_height_px)

    def render_chunk(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
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

        renderer = self.renderer
        if not renderer:
            return

        yield from renderer.render_chunk(
            self,
            width,
            height,
            max_chunk_width=max_chunk_width,
            max_chunk_height=max_chunk_height,
            max_memory_size=max_memory_size,
        )

    def get_tab_direction(self, tab: Tab) -> Optional[Tuple[float, float]]:
        """
        Calculates the "outside" direction vector for a given tab in world
        coordinates.

        The direction is a normalized 2D vector representing the outward
        normal of the geometry at the tab's location, transformed by the
        workpiece's rotation and scaling.

        Args:
            tab: The Tab object for which to find the direction.

        Returns:
            A tuple (dx, dy) representing the direction vector, or None if
            the workpiece has no vector data or the path is open.
        """
        if self.vectors is None:
            return None

        # 1. Get the normal vector in the geometry's local space.
        local_normal = self.vectors.get_outward_normal_at(
            tab.segment_index, tab.pos
        )
        if local_normal is None:
            return None

        # For non-uniform scaling, the normal must be transformed by the
        # inverse transpose of the world matrix to remain perpendicular.
        world_matrix_3x3 = self.get_world_transform().to_numpy()
        try:
            # Get the top-left 2x2 part for the normal transformation
            m_2x2 = world_matrix_3x3[:2, :2]
            m_inv_T = np.linalg.inv(m_2x2).T
            transformed_vector = m_inv_T @ np.array(local_normal)
        except np.linalg.LinAlgError:
            # Fallback for non-invertible matrices (e.g., zero scale)
            return self.get_world_transform().transform_vector(local_normal)

        tx, ty = transformed_vector
        norm = math.sqrt(tx**2 + ty**2)
        if norm < 1e-9:
            return (1.0, 0.0)  # Fallback

        return (tx / norm, ty / norm)

    def dump(self, indent=0):
        source_file = self.source_file
        renderer = self.renderer
        renderer_name = renderer.__class__.__name__ if renderer else "None"
        print("  " * indent, source_file, renderer_name)

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
