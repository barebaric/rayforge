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
)
from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from ..core.ops import Ops
from .item import DocItem
from .matrix import Matrix

if TYPE_CHECKING:
    from .layer import Layer
    from ..importer.base_renderer import Renderer


logger = logging.getLogger(__name__)


class WorkPiece(DocItem):
    """
    Represents a real-world workpiece. It is a lightweight data container,
    holding its source data (_data, source_ops), its transformation matrix,
    and a Renderer for display. It is completely decoupled from importers.
    """

    def __init__(
        self,
        source_file: Path,
        renderer: "Renderer",
        data: Optional[bytes] = None,
        source_ops: Optional[Ops] = None,
    ):
        super().__init__(name=source_file.name)
        self.source_file = source_file
        self.renderer = renderer
        self._data = data
        self.source_ops = source_ops

        # The cache for rendered vips images. Key is (width, height).
        # This is the proper place for this state, not monkey-patched.
        self._render_cache: Dict[Tuple[int, int], pyvips.Image] = {}

    def clear_render_cache(self):
        """
        Invalidates and clears all cached renders for this workpiece.
        Should be called if the underlying _data or source_ops changes.
        """
        self._render_cache.clear()

    @property
    def data(self) -> Optional[bytes]:
        return self._data

    @data.setter
    def data(self, new_data: Optional[bytes]):
        if self._data != new_data:
            self._data = new_data
            self.clear_render_cache()
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
        document hierarchy.
        """
        # Create a new instance to avoid side effects with signals,
        # parents, etc.
        world_wp = WorkPiece(
            self.source_file, self.renderer, self._data, self.source_ops
        )
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a dictionary.
        """
        return {
            "uid": self.uid,
            "name": self.name,
            "matrix": self._matrix.to_list(),
            "renderer_name": self.renderer.__class__.__name__,
            "source_ops": self.source_ops.to_dict()
            if self.source_ops
            else None,
            "data": self._data,
            "source_file": str(self.source_file),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "WorkPiece":
        """
        Restores a WorkPiece instance from a dictionary.
        """
        from ..importer import renderer_by_name
        from ..core.ops import Ops

        renderer = renderer_by_name[state["renderer_name"]]
        source_file = Path(state["source_file"])
        source_ops = (
            Ops.from_dict(state["source_ops"]) if state["source_ops"] else None
        )

        wp = cls(
            source_file=source_file,
            renderer=renderer,
            data=state["data"],
            source_ops=source_ops,
        )
        wp.uid = state["uid"]
        wp.name = state["name"]
        wp.matrix = Matrix.from_list(state["matrix"])
        return wp

    def get_natural_size(self) -> Optional[Tuple[float, float]]:
        return self.renderer.get_natural_size(self)

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
        return self.renderer.render_to_pixels(self, width, height)

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

        return self.renderer.render_to_pixels(
            self, target_width_px, target_height_px
        )

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

        yield from self.renderer.render_chunk(
            self,
            width,
            height,
            max_chunk_width=max_chunk_width,
            max_chunk_height=max_chunk_height,
            max_memory_size=max_memory_size,
        )

    def dump(self, indent=0):
        print(
            "  " * indent, self.source_file, self.renderer.__class__.__name__
        )

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
