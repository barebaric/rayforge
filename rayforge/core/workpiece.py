from __future__ import annotations
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
    NamedTuple,
    Union,
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

from ..context import get_context
from .geo import Geometry
from .item import DocItem
from .matrix import Matrix
from .source_asset_segment import SourceAssetSegment
from .tab import Tab

if TYPE_CHECKING:
    from ..image.base_renderer import Renderer
    from .asset import IAsset
    from .layer import Layer
    from .source_asset import SourceAsset
    from .sketcher.sketch import Sketch


logger = logging.getLogger(__name__)

CAIRO_MAX_DIMENSION = 16384


class RenderContext(NamedTuple):
    """Encapsulates the resources required for rendering."""

    data: Union[bytes, Any]
    renderer: "Renderer"
    source_pixel_dims: Optional[Tuple[int, int]]
    metadata: Dict[str, Any]


class WorkPiece(DocItem):
    """
    Represents a real-world workpiece. It is a lightweight data container,
    holding its transformation matrix and a link to its source and shape
    definition.
    """

    def __init__(
        self,
        name: str,
        source_segment: Optional[SourceAssetSegment] = None,
    ):
        super().__init__(name=name)
        self._source_segment = source_segment
        self._boundaries_cache: Optional[Geometry] = None

        # Natural (untransformed) dimensions of the workpiece content.
        self.natural_width_mm: float = 0.0
        self.natural_height_mm: float = 0.0

        # An optional override for the workpiece geometry. If set, this takes
        # precedence over the source_segment's geometry. It allows for
        # non-destructive editing (like splitting) without modifying the
        # shared source segment.
        self._edited_boundaries: Optional[Geometry] = None

        # The cache for rendered vips images. Key is (width, height).
        self._render_cache: Dict[Tuple[int, int], pyvips.Image] = {}

        # Transient attributes for deserialized instances in subprocesses
        self._data: Optional[bytes] = None
        self._original_data: Optional[bytes] = None
        self._renderer: Optional["Renderer"] = None
        self._transient_source_px_dims: Optional[Tuple[int, int]] = None

        # Parametric sketch fields
        self.sketch_uid: Optional[str] = None
        self._sketch_params: Dict[str, Any] = {}
        self._transient_sketch_definition: Optional["Sketch"] = None

        self._tabs: List[Tab] = []
        self._tabs_enabled: bool = True

        # Transient cache for UI view artifacts (Cairo surfaces, etc.)
        # This persists across view element destruction/creation
        # (e.g. Grouping) but is not serialized to disk.
        self._view_cache: Dict[str, Any] = {}

    def depends_on_asset(self, asset: "IAsset") -> bool:
        """
        Checks if this workpiece depends on the given asset, either through
        its sketch definition or its source file.
        """
        if self.sketch_uid and self.sketch_uid == asset.uid:
            return True
        if (
            self.source_segment
            and self.source_segment.source_asset_uid == asset.uid
        ):
            return True
        return False

    @classmethod
    def from_sketch(cls, sketch: "Sketch") -> "WorkPiece":
        """
        Factory method to create a WorkPiece from a Sketch definition.

        This method performs a transient solve of the sketch to determine
        its natural dimensions and initialize the WorkPiece's transformation
        matrix correctly before it is added to the document.
        """
        from .sketcher.sketch import (
            Sketch,
        )  # Lazy import to avoid circular dep

        # 1. Solve a transient copy to determine natural size without side
        # effects.
        geometry = None
        min_x, min_y = 0.0, 0.0

        try:
            temp_sketch = Sketch.from_dict(sketch.to_dict())
            temp_sketch.solve()
            geometry = temp_sketch.to_geometry()

            if not geometry.is_empty():
                min_x, min_y, max_x, max_y = geometry.rect()
                width = max(max_x - min_x, 1e-9)
                height = max(max_y - min_y, 1e-9)
            else:
                width, height = 1.0, 1.0
        except Exception as e:
            logger.warning(
                f"Failed to calculate initial geometry for sketch "
                f"{sketch.uid}: {e}"
            )
            width, height = 1.0, 1.0
            from .geo import Geometry

            geometry = Geometry()

        # 2. Create the instance
        instance = cls(name=sketch.name or "Sketch")
        instance.sketch_uid = sketch.uid
        instance.natural_width_mm = width
        instance.natural_height_mm = height

        # 3. Set the transformation matrix scale to match natural size.
        if width > 1e-6 and height > 1e-6:
            instance.set_size(width, height)

        # 4. Pre-populate boundaries cache to avoid immediate re-solve on
        # first render. We perform the same normalization here that the
        # boundaries property does.
        if geometry and not geometry.is_empty():
            norm_matrix = Matrix.scale(
                1.0 / width, 1.0 / height
            ) @ Matrix.translation(-min_x, -min_y)
            geometry.transform(norm_matrix.to_4x4_numpy())

        # Cache the result (even if empty) to ensure fast rendering
        instance._boundaries_cache = geometry

        return instance

    @property
    def source_segment(self) -> Optional[SourceAssetSegment]:
        """The source data definition for this workpiece."""
        return self._source_segment

    @source_segment.setter
    def source_segment(self, new_segment: Optional[SourceAssetSegment]):
        """
        Sets a new source segment, clearing caches and signaling an update.
        This is the correct way to modify a workpiece's source data.
        """
        if self._source_segment != new_segment:
            self._source_segment = new_segment
            # Invalidate all cached data that depends on the source.
            self.clear_render_cache()
            # Signal that the workpiece's content has changed. This is crucial
            # for triggering the pipeline to re-process the geometry.
            self.updated.send(self)

    @property
    def natural_size(self) -> Tuple[float, float]:
        """
        Returns the natural (untransformed) size of the content in mm.
        This is the authoritative source for the workpiece's intrinsic size.
        """
        return (self.natural_width_mm, self.natural_height_mm)

    def get_local_bbox(self) -> Optional[Tuple[float, float, float, float]]:
        """
        WorkPieces are geometrically defined as a unit square (0,0,1,1) that is
        scaled by their matrix.
        """
        return (0.0, 0.0, 1.0, 1.0)

    def clear_render_cache(self):
        """
        Invalidates and clears all cached renders for this workpiece.
        Should be called if the underlying data or geometry changes.
        """
        logger.debug(
            f"WP {self.uid[:8]}: Clearing all caches (render and boundaries)."
        )
        self._render_cache.clear()
        self._boundaries_cache = (
            None  # Also clear the boundaries cache on update
        )

    @property
    def source(self) -> "Optional[SourceAsset]":
        """
        Convenience property to retrieve the full SourceAsset object from the
        document's central registry.
        """
        if self.doc and self.source_segment:
            return self.doc.get_source_asset_by_uid(
                self.source_segment.source_asset_uid
            )
        return None

    @property
    def original_data(self) -> Optional[bytes]:
        """
        Retrieves the original, unmodified data from the source asset.
        """
        # Prioritize transient data for isolated/subprocess instances
        if self._original_data is not None:
            return self._original_data
        source = self.source
        return source.original_data if source else None

    @property
    def data(self) -> Optional[bytes]:
        """
        Retrieves the appropriate source data for rendering.

        This property intelligently selects the correct data:
        1. Prioritizes transient data for isolated/subprocess instances.
        2. Prioritizes the `base_render_data` (e.g., a trimmed SVG or
           cropped PDF) if it exists, as this is what the workpiece's size
           is based on.
        3. Falls back to the `original_data` if no specific render data
           is available.
        """
        # Prioritize transient data for isolated/subprocess instances
        if self._data is not None:
            return self._data
        source = self.source
        if not source:
            return None

        # Prioritize the processed render data if it exists.
        if source.base_render_data is not None:
            return source.base_render_data

        # Fall back to the original data.
        return source.original_data

    @property
    def source_file(self) -> Optional[Path]:
        """Retrieves the source file path from the linked SourceAsset."""
        source = self.source
        return source.source_file if source else None

    @property
    def _active_renderer(self) -> "Optional[Renderer]":
        """Retrieves the renderer (internal use)."""
        if self._renderer is not None:
            return self._renderer

        # If it's a sketch, we know the renderer.
        if self.sketch_uid:
            from ..image.sketch.renderer import SKETCH_RENDERER

            return SKETCH_RENDERER

        source = self.source
        return source.renderer if source else None

    @property
    def boundaries(self) -> Optional[Geometry]:
        """
        The normalized vector geometry defining the workpiece shape.

        This `Geometry` object represents the workpiece's intrinsic shape,
        normalized to fit within a 1x1 unit reference box. This separation of
        intrinsic shape from its world transformation is crucial for
        preventing rendering and processing errors.

        If `_edited_boundaries` is set (e.g. from a split operation), it is
        returned. Otherwise, the geometry from the `source_segment` is used.

        The local coordinate space of this normalized geometry has the
        following properties:

        - **Coordinate System**: Y-up, where (0, 0) is the bottom-left corner.
        - **Reference Size**: The geometry is scaled to fit within a box
          that is 1 unit wide by 1 unit tall.
        - **Origin (0,0)**: The anchor point is the bottom-left corner of the
          geometry's bounding box.
        - **Transformation**: The vector data itself is static. All physical
          sizing, positioning, and rotation are handled by applying the
          `WorkPiece.matrix` to this normalized shape.
        """
        logger.debug("boundaries called")
        if self._edited_boundaries is not None:
            return self._edited_boundaries

        if self._boundaries_cache is not None:
            logger.debug("Cache hit: boundaries present")
            return self._boundaries_cache
        logger.debug("Cache miss: boundaries not present")

        # --- Sketch-based Geometry Generation ---
        if self.sketch_uid:
            sketch_def = self.get_sketch_definition()
            if not sketch_def:
                return None

            from .sketcher.sketch import Sketch

            instance_sketch = Sketch.from_dict(sketch_def.to_dict())
            for key, val in self.sketch_params.items():
                if key in instance_sketch.input_parameters.keys():
                    try:
                        instance_sketch.input_parameters[key] = val
                    except Exception:
                        pass

            instance_sketch.solve()
            unnormalized_geo = instance_sketch.to_geometry()

            # Cache the geometry even if it is empty, to prevent
            # re-solving on every render frame.
            if unnormalized_geo.is_empty():
                self._boundaries_cache = unnormalized_geo
                return self._boundaries_cache

            # Normalize the solved geometry to a 0-1 box (Y-Up)
            min_x, min_y, max_x, max_y = unnormalized_geo.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)
            norm_matrix = Matrix.scale(
                1.0 / width, 1.0 / height
            ) @ Matrix.translation(-min_x, -min_y)
            unnormalized_geo.transform(norm_matrix.to_4x4_numpy())

            self._boundaries_cache = unnormalized_geo
            return self._boundaries_cache

        # --- SourceAssetSegment-based Geometry (legacy/other formats) ---
        if not self.source_segment:
            logger.warning(
                f"WP {self.uid[:8]}: Cannot get boundaries, no source_segment."
            )
            return None

        # Get the Y-down geometry from storage
        y_down_geo = self.source_segment.segment_mask_geometry
        is_geom_empty = not y_down_geo or y_down_geo.is_empty()
        logger.debug(
            f"WP {self.uid[:8]}: Recalculating boundaries from segment. "
            f"Stored geometry is_empty={is_geom_empty}"
        )
        if is_geom_empty:
            return None

        y_up_geo = y_down_geo.copy()

        # Transformation to flip Y axis in a 0-1 box:
        # 1. Scale Y by -1 (flips it from Y-down to Y-up)
        # 2. Translate Y by +1 (moves it back into the 0-1 box)
        flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
        y_up_geo.transform(flip_matrix.to_4x4_numpy())

        logger.debug(
            f"WP {self.uid[:8]}: Caching new boundaries. "
            f"Rect: {y_up_geo.rect()}"
        )
        self._boundaries_cache = y_up_geo
        return self._boundaries_cache

    @property
    def _boundaries_y_down(self) -> Optional[Geometry]:
        """
        Internal helper to get the appropriate Y-DOWN normalized geometry
        for use in image masking, which operates in a Y-down pixel space.
        """
        # Case 1: An edit (like a split) has occurred. The authoritative
        # geometry is `_edited_boundaries`, which is Y-UP. We must flip it.
        if self._edited_boundaries is not None:
            y_down_geo = self._edited_boundaries.copy()
            # Flip Y-up (0,0 at bottom) to Y-down (0,0 at top)
            flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            y_down_geo.transform(flip_matrix.to_4x4_numpy())
            return y_down_geo

        # For sketches, we must generate from the Y-up boundaries
        if self.sketch_uid:
            y_up_geo = self.boundaries
            if not y_up_geo:
                return None
            y_down_geo = y_up_geo.copy()
            flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            y_down_geo.transform(flip_matrix.to_4x4_numpy())
            return y_down_geo

        if self.source_segment:
            return self.source_segment.segment_mask_geometry

        return None

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
        world_wp = WorkPiece(self.name, deepcopy(self.source_segment))
        world_wp.uid = self.uid
        world_wp.matrix = self.get_world_transform()
        world_wp.tabs = deepcopy(self.tabs)
        world_wp.tabs_enabled = self.tabs_enabled
        world_wp.sketch_uid = self.sketch_uid
        world_wp.sketch_params = deepcopy(self._sketch_params)

        # Ensure any edited boundaries are carried over.
        if self._edited_boundaries:
            world_wp._edited_boundaries = self._edited_boundaries.copy()

        # Hydrate with data and renderer for use in isolated contexts
        # like subprocesses where the document link is lost.
        source = self.source
        if source:
            world_wp._data = self.data
            world_wp._original_data = self.original_data
            world_wp._renderer = source.renderer
            if source.width_px is not None and source.height_px is not None:
                world_wp._transient_source_px_dims = (
                    source.width_px,
                    source.height_px,
                )

        # Hydrate the transient sketch definition if it exists
        if self.sketch_uid and self.doc:
            sketch = self.doc.get_asset_by_uid(self.sketch_uid)
            if sketch:
                # We copy the sketch definition so the subprocess has
                # an independent object
                from .sketcher.sketch import Sketch

                world_wp._transient_sketch_definition = Sketch.from_dict(
                    sketch.to_dict()
                )

        return world_wp

    def get_sketch_definition(self) -> Optional["Sketch"]:
        """
        Retrieves the Sketch definition for this workpiece, if applicable.
        Prioritizes the transient definition (for subprocesses), then checks
        the document registry.
        """
        if self._transient_sketch_definition:
            return self._transient_sketch_definition
        if self.sketch_uid and self.doc:
            return cast(
                Optional["Sketch"], self.doc.get_asset_by_uid(self.sketch_uid)
            )
        return None

    def _resolve_render_context(self) -> Optional[RenderContext]:
        """
        Resolves the data, renderer, and metadata needed for rendering.
        Unifies logic for transient (subprocess) vs. managed (document) states.
        """
        # For sketches, the "data" is not needed, as the geometry is generated
        # by the `boundaries` property. We pass empty bytes.
        if self.sketch_uid:
            from ..image.sketch.renderer import SKETCH_RENDERER

            return RenderContext(
                data=b"",
                renderer=SKETCH_RENDERER,
                source_pixel_dims=None,
                metadata={"is_vector": True},
            )

        # --- Fallback to standard SourceAsset logic ---

        # 1. Renderer
        renderer = self._active_renderer
        if not renderer:
            return None

        # 2. Data
        data_to_render = self.data
        if data_to_render is None:
            return None

        # 3. Source Pixel Dimensions
        source_px_dims = self._transient_source_px_dims
        if not source_px_dims and self.source:
            if (
                self.source.width_px is not None
                and self.source.height_px is not None
            ):
                source_px_dims = (
                    self.source.width_px,
                    self.source.height_px,
                )

        # 4. Metadata
        metadata = self.source.metadata if self.source else {}

        return RenderContext(
            data=data_to_render,
            renderer=renderer,
            source_pixel_dims=source_px_dims,
            metadata=metadata,
        )

    def _calculate_render_geometry(
        self,
        target_width: int,
        target_height: int,
        source_px_dims: Optional[Tuple[int, int]],
    ) -> Tuple[int, int, Optional[Tuple[int, int, int, int]], Optional[bytes]]:
        """
        Calculates the required render dimensions and optional crop rectangle.
        Handles the logic for high-res rendering of cropped segments.

        Returns:
            (render_width, render_height, crop_rect, data_override)
            - render_width/height: The size to ask the renderer for.
            - crop_rect: The (x, y, w, h) tuple for post-render cropping,
              or None.
            - data_override: Specific bytes to use if switching to
              original_data is required (e.g. for cropping), or None to
              use default.
        """
        render_width, render_height = target_width, target_height
        crop_rect = None
        data_override = None

        is_cropped = (
            self.source_segment
            and self.source_segment.crop_window_px is not None
        )

        # If cropping, we must render the *original* full image at a scaled-up
        # resolution such that the crop window matches the target dimensions.
        if is_cropped and self.original_data and source_px_dims:
            # We need to switch to the original data for the full render
            data_override = self.original_data
            source_w, source_h = source_px_dims

            if self.source_segment and self.source_segment.crop_window_px:
                crop_x_f, crop_y_f, crop_w_f, crop_h_f = (
                    self.source_segment.crop_window_px
                )
                crop_w = float(crop_w_f)
                crop_h = float(crop_h_f)

                if crop_w > 0 and crop_h > 0:
                    scale_x = target_width / crop_w
                    scale_y = target_height / crop_h
                    render_width = max(1, int(source_w * scale_x))
                    render_height = max(1, int(source_h * scale_y))

                    # Calculate the expected crop rectangle in the scaled
                    # image.
                    # We return this as integers for the crop function.
                    # Note: We recalculate this precisely based on the *actual*
                    # rendered image size in _process_rendered_image to handle
                    # renderer rounding, but this gives us the intent.
                    scaled_x = int(crop_x_f * scale_x)
                    scaled_y = int(crop_y_f * scale_y)
                    scaled_w = int(crop_w * scale_x)
                    scaled_h = int(crop_h * scale_y)
                    crop_rect = (scaled_x, scaled_y, scaled_w, scaled_h)

        return render_width, render_height, crop_rect, data_override

    def _build_renderer_kwargs(
        self, renderer: "Renderer", metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Constructs format-specific arguments for the renderer."""
        kwargs = {}
        renderer_name = renderer.__class__.__name__

        if renderer_name in ("OpsRenderer", "DxfRenderer", "SketchRenderer"):
            kwargs["boundaries"] = self.boundaries

        if renderer_name == "DxfRenderer":
            kwargs["source_metadata"] = metadata
            kwargs["workpiece_matrix"] = self.matrix

        return kwargs

    def _process_rendered_image(
        self,
        image: pyvips.Image,
        crop_rect_hint: Optional[Tuple[int, int, int, int]],
        target_size: Tuple[int, int],
        source_px_dims: Optional[Tuple[int, int]],
    ) -> Optional[pyvips.Image]:
        """
        Applies post-processing steps: cropping, masking, and final resizing.
        """
        from ..image import image_util

        processed_image = image
        target_w, target_h = target_size

        # 1. Apply Crop
        # We re-calculate the crop rect based on the actual image dimensions
        # to be robust against renderers that might snap to nearest DPI/size.
        if (
            crop_rect_hint
            and self.source_segment
            and self.source_segment.crop_window_px
        ):
            crop_x, crop_y, crop_w, crop_h = map(
                float, self.source_segment.crop_window_px
            )
            actual_w = processed_image.width
            actual_h = processed_image.height

            # Re-derive scale from actual output vs assumed source
            scale_x = 1.0
            scale_y = 1.0
            if source_px_dims:
                if source_px_dims[0] > 0:
                    scale_x = actual_w / source_px_dims[0]
                if source_px_dims[1] > 0:
                    scale_y = actual_h / source_px_dims[1]

            scaled_x = int(crop_x * scale_x)
            scaled_y = int(crop_y * scale_y)
            scaled_w = int(crop_w * scale_x)
            scaled_h = int(crop_h * scale_y)

            processed_image = image_util.safe_crop(
                processed_image, scaled_x, scaled_y, scaled_w, scaled_h
            )
            if not processed_image:
                return None

        # 2. Apply Mask
        # We skip masking for Vector sources because they already render with
        # correct transparency, and masking with vector geometry (which can
        # be open lines with zero area) would incorrectly hide the content.
        is_vector = False
        if self.source and self.source.metadata.get("is_vector"):
            is_vector = True

        # Special check for Sketch rendering: if we rendered via sketch def,
        # it is vector data.
        if self.sketch_uid:
            is_vector = True

        if not is_vector:
            mask_geo = self._boundaries_y_down
            if mask_geo and not mask_geo.is_empty():
                processed_image = image_util.apply_mask_to_vips_image(
                    processed_image, mask_geo
                )
                if not processed_image:
                    return None

        # 3. Final Resize Check
        if (
            processed_image.width != target_w
            or processed_image.height != target_h
        ):
            if processed_image.width > 0 and processed_image.height > 0:
                h_scale = target_w / processed_image.width
                v_scale = target_h / processed_image.height
                processed_image = processed_image.resize(
                    h_scale, vscale=v_scale
                )

        return processed_image

    def get_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        """
        The central hub for rendering a vips image for this workpiece.
        Orchestrates data retrieval, rendering, cropping, and masking.
        """
        key = (width, height)
        if key in self._render_cache:
            return self._render_cache[key]

        # 1. Resolve Context
        ctx = self._resolve_render_context()
        if not ctx:
            logger.warning(
                f"WP {self.uid[:8]}: Could not resolve render context."
            )
            return None

        # 2. Calculate Geometry
        (
            render_w,
            render_h,
            crop_rect_hint,
            data_override,
        ) = self._calculate_render_geometry(
            width, height, ctx.source_pixel_dims
        )
        final_data = data_override if data_override else ctx.data

        # 3. Build Config
        kwargs = self._build_renderer_kwargs(ctx.renderer, ctx.metadata)

        # 4. Render
        raw_image = ctx.renderer.render_base_image(
            final_data, render_w, render_h, **kwargs
        )
        if not raw_image:
            logger.warning(f"WP {self.uid[:8]}: Renderer returned None.")
            return None

        # 5. Process (Crop/Mask/Resize)
        final_image = self._process_rendered_image(
            raw_image, crop_rect_hint, (width, height), ctx.source_pixel_dims
        )

        if final_image:
            self._render_cache[key] = final_image

        return final_image

    def get_local_size(self) -> Tuple[float, float]:
        """
        The local-space size (width, height) in mm, as absolute values,
        decomposed from the local transformation matrix. This is used for
        determining rasterization resolution.
        """
        return self.matrix.get_abs_scale()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the WorkPiece state to a dictionary. Includes transient
        data if it has been hydrated.
        """
        state = {
            "uid": self.uid,
            "name": self.name,
            "matrix": self._matrix.to_list(),
            "width_mm": self.natural_width_mm,
            "height_mm": self.natural_height_mm,
            "tabs": [asdict(t) for t in self._tabs],
            "tabs_enabled": self._tabs_enabled,
            "source_segment": (
                self.source_segment.to_dict() if self.source_segment else None
            ),
            "edited_boundaries": (
                self._edited_boundaries.to_dict()
                if self._edited_boundaries
                else None
            ),
            "sketch_uid": self.sketch_uid,
            "sketch_params": self._sketch_params,
        }
        # Include hydrated data for subprocesses
        if self._data is not None:
            state["data"] = self._data
        if self._original_data is not None:
            state["original_data"] = self._original_data
        if self._renderer is not None:
            state["renderer_name"] = self._renderer.__class__.__name__
        if self._transient_source_px_dims is not None:
            state["source_px_dims"] = self._transient_source_px_dims
        if self._transient_sketch_definition is not None:
            state["transient_sketch_definition"] = (
                self._transient_sketch_definition.to_dict()
            )
        return state

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPiece":
        """
        Restores a WorkPiece instance from a dictionary.
        """
        config_data = data.get("source_segment")
        source_segment = (
            SourceAssetSegment.from_dict(config_data) if config_data else None
        )

        wp = cls(
            name=data["name"],
            source_segment=source_segment,
        )
        wp.uid = data["uid"]
        wp.matrix = Matrix.from_list(data["matrix"])
        wp.natural_width_mm = data.get("width_mm", 0.0)
        wp.natural_height_mm = data.get("height_mm", 0.0)

        loaded_tabs = []
        for t_data in data.get("tabs", []):
            t_data_copy = t_data.copy()
            # Ignore 'length' for backward compatibility with older files.
            t_data_copy.pop("length", None)
            loaded_tabs.append(Tab(**t_data_copy))
        wp.tabs = loaded_tabs
        wp.tabs_enabled = data.get("tabs_enabled", True)

        if "edited_boundaries" in data and data["edited_boundaries"]:
            wp._edited_boundaries = Geometry.from_dict(
                data["edited_boundaries"]
            )

        wp.sketch_uid = data.get("sketch_uid")
        wp._sketch_params = data.get("sketch_params", {})

        # Hydrate with transient data if provided for subprocesses
        if "data" in data:
            wp._data = data["data"]
        if "original_data" in data:
            wp._original_data = data["original_data"]
        if "source_px_dims" in data:
            wp._transient_source_px_dims = tuple(data["source_px_dims"])
        if "renderer_name" in data:
            renderer_name = data["renderer_name"]
            from ..image import renderer_by_name

            if renderer_name in renderer_by_name:
                wp._renderer = renderer_by_name[renderer_name]

        if "transient_sketch_definition" in data:
            from .sketcher.sketch import Sketch

            wp._transient_sketch_definition = Sketch.from_dict(
                data["transient_sketch_definition"]
            )
        if "sketch_params" in data:
            wp._sketch_params = data.get("sketch_params", {})
        else:
            wp._sketch_params = data.get("_sketch_params", {})

        return wp

    @property
    def sketch_params(self) -> Dict[str, Any]:
        """Get the sketch parameters for this workpiece."""
        return self._sketch_params

    @sketch_params.setter
    def sketch_params(self, new_params: Dict[str, Any]):
        """
        Set the sketch parameters and trigger regeneration if needed.
        """
        if self._sketch_params != new_params:
            self._sketch_params = new_params
            if self.sketch_uid:
                self.regenerate_from_sketch()

    def natural_sizes(self) -> Optional[Tuple[float, float]]:
        """
        Legacy method alias. Use property `natural_size` instead.
        """
        return self.natural_size

    def get_natural_aspect_ratio(self) -> Optional[float]:
        size = self.natural_size
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
        size = self.natural_size
        if size and size[0] > 0 and size[1] > 0:
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

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        from ..image import image_util

        # This now uses the central hub for rendering.
        final_image = self.get_vips_image(width, height)
        if not final_image:
            return None
        normalized_image = image_util.normalize_to_rgba(final_image)
        if not normalized_image:
            return None
        return image_util.vips_rgba_to_cairo_surface(normalized_image)

    def render_for_ops(
        self,
        pixels_per_mm_x: float,
        pixels_per_mm_y: float,
    ) -> Optional[cairo.ImageSurface]:
        """Renders to a pixel surface at the workpiece's current size.
        Returns None if size is not valid."""
        # Use the final world-space size for rendering resolution.
        current_size = self.size
        if not current_size or current_size[0] <= 0 or current_size[1] <= 0:
            return None

        width_mm, height_mm = current_size
        target_width_px = int(width_mm * pixels_per_mm_x)
        target_height_px = int(height_mm * pixels_per_mm_y)

        return self.render_to_pixels(target_width_px, target_height_px)

    def _calculate_chunk_layout(
        self,
        real_width: int,
        real_height: int,
        max_chunk_width: Optional[int],
        max_chunk_height: Optional[int],
        max_memory_size: Optional[int],
    ) -> Tuple[int, int, int, int]:
        bytes_per_pixel = 4
        effective_max_width = min(
            max_chunk_width
            if max_chunk_width is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )
        chunk_width = min(real_width, effective_max_width)
        possible_heights = []
        effective_max_height = min(
            max_chunk_height
            if max_chunk_height is not None
            else CAIRO_MAX_DIMENSION,
            CAIRO_MAX_DIMENSION,
        )
        possible_heights.append(effective_max_height)
        if max_memory_size is not None and chunk_width > 0:
            height_from_mem = math.floor(
                max_memory_size / (chunk_width * bytes_per_pixel)
            )
            possible_heights.append(height_from_mem)
        chunk_height = min(real_height, *possible_heights)
        chunk_width = max(1, chunk_width)
        chunk_height = max(1, chunk_height)
        cols = math.ceil(real_width / chunk_width)
        rows = math.ceil(real_height / chunk_height)
        return chunk_width, cols, chunk_height, rows

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
        from ..image import image_util

        # Use the final world-space size for rendering resolution.
        current_size = self.size
        if not current_size or current_size[0] <= 0 or current_size[1] <= 0:
            return

        width_px = current_size[0] * pixels_per_mm_x
        height_px = current_size[1] * pixels_per_mm_y

        if all(
            arg is None
            for arg in [max_chunk_width, max_chunk_height, max_memory_size]
        ):
            raise ValueError(
                "At least one of max_chunk_width, max_chunk_height, "
                "or max_memory_size must be provided."
            )

        vips_image = self.get_vips_image(round(width_px), round(height_px))
        if not vips_image or not isinstance(vips_image, pyvips.Image):
            logger.warning("Failed to load image for chunking.")
            return

        real_width = cast(int, vips_image.width)
        real_height = cast(int, vips_image.height)
        if not real_width or not real_height:
            return

        chunk_width, cols, chunk_height, rows = self._calculate_chunk_layout(
            real_width,
            real_height,
            max_chunk_width,
            max_chunk_height,
            max_memory_size,
        )

        overlap_x, overlap_y = 1, 0  # Default overlap values

        for row in range(rows):
            for col in range(cols):
                left = col * chunk_width
                top = row * chunk_height
                width = min(chunk_width + overlap_x, real_width - left)
                height = min(chunk_height + overlap_y, real_height - top)

                if width <= 0 or height <= 0:
                    continue

                chunk: pyvips.Image = vips_image.crop(left, top, width, height)

                normalized_chunk = image_util.normalize_to_rgba(chunk)
                if not normalized_chunk:
                    logger.warning(
                        f"Could not normalize chunk at ({left},{top})"
                    )
                    continue

                surface = image_util.vips_rgba_to_cairo_surface(
                    normalized_chunk
                )
                yield surface, (left, top)

    def get_geometry_world_bbox(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the bounding box of the workpiece's geometry in world
        coordinates.

        This is achieved by creating a temporary copy of the geometry,
        transforming it by the workpiece's world matrix, and then calculating
        the bounding box of the transformed shape.

        Returns:
            A tuple (min_x, min_y, max_x, max_y) representing the bounding
            box, or None if the workpiece has no vector geometry.
        """
        boundaries = self.boundaries
        if boundaries is None or boundaries.is_empty():
            return None

        # Create a copy to avoid modifying the original normalized vectors
        world_geometry = boundaries.copy()

        # Apply the full world transformation
        world_matrix = self.get_world_transform()
        world_geometry.transform(world_matrix.to_4x4_numpy())

        # Return the bounding box of the transformed geometry
        return world_geometry.rect()

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
        boundaries = self.boundaries
        if boundaries is None:
            return None

        # 1. Get the normal vector in the geometry's local space.
        local_normal = boundaries.get_outward_normal_at(
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
        renderer = self._active_renderer
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

        context = get_context()
        if not context.config or not context.machine:
            return None

        machine = context.machine
        model_x, model_y = current_pos  # Canonical: Y-up, bottom-left

        if machine.y_axis_down:
            # Convert to machine: Y-down, top-left
            machine_height = machine.dimensions[1]
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

        context = get_context()
        if not context.config or not context.machine:
            return

        machine = context.machine
        machine_x, machine_y = pos
        model_pos = (0.0, 0.0)

        if machine.y_axis_down:
            # Convert from machine (Y-down, top-left) to
            # model (Y-up, bottom-left)
            machine_height = machine.dimensions[1]
            model_y = machine_height - machine_y - current_size[1]
            model_pos = machine_x, model_y
        else:
            # Machine is Y-up, same as model
            model_pos = machine_x, machine_y

        self.pos = model_pos

    def apply_split(self, fragments: List[Geometry]) -> List["WorkPiece"]:
        """
        Creates new WorkPiece instances from a list of normalized geometry
        fragments. Each fragment represents a subset of this workpiece's
        current geometry.

        The new workpieces will have their own independent SourceAssetSegment
        containing only the specific fragment geometry.

        Args:
            fragments: A list of Geometry objects. Each must be a subset of
                       self.boundaries, defined in the same 0-1 Y-up
                       normalized coordinate space.

        Returns:
            A list of new WorkPiece instances.
        """
        if len(fragments) <= 1:
            return [self]

        new_workpieces = []
        original_matrix = self.matrix

        # Get current physical dimensions to filter noise.
        # self.size returns (width_mm, height_mm).
        phys_w, phys_h = self.size

        for frag_geo in fragments:
            # 1. Calculate bounding box of the fragment in the local 0-1 space.
            min_x, min_y, max_x, max_y = frag_geo.rect()
            w = max(max_x - min_x, 1e-9)
            h = max(max_y - min_y, 1e-9)

            # 2. Filter out noise / dust.
            # Calculate physical dimensions of the fragment.
            # Fragments smaller than 0.1mm in both dimensions are discarded
            # to prevent creating hundreds of invisible workpieces that clog
            # the renderer and UI.
            if (w * phys_w < 0.1) and (h * phys_h < 0.1):
                continue

            # 3. Normalize the fragment geometry.
            # We shift it to (0,0) and scale it to fit a 1x1 box.
            # This becomes the new canonical shape for this piece.
            normalized_frag = frag_geo.copy()
            norm_matrix = Matrix.scale(1.0 / w, 1.0 / h) @ Matrix.translation(
                -min_x, -min_y
            )
            normalized_frag.transform(norm_matrix.to_4x4_numpy())

            # 4. Create the new segment using the cleaner API.
            new_segment = None
            if self.source_segment:
                # Convert Y-up to Y-down format expected by storage
                y_down_frag = normalized_frag.copy()
                flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
                y_down_frag.transform(flip_matrix.to_4x4_numpy())

                # The clone method to efficiently creates a separate segment
                new_segment = self.source_segment.clone_with_geometry(
                    y_down_frag
                )

            # Initialize the new workpiece with the lightweight segment
            new_wp = WorkPiece(self.name, new_segment)
            new_wp.tabs_enabled = self.tabs_enabled

            # 5. Calculate the matrix for the new piece.
            # It must be positioned such that it aligns with where this
            # fragment was in the original object.
            # Matrix op order: parent @ child.
            # Compose: M_orig @ T_local_offset @ S_local_scale.
            offset_matrix = original_matrix @ Matrix.translation(min_x, min_y)
            final_matrix = offset_matrix @ Matrix.scale(w, h)

            new_wp.matrix = final_matrix
            new_workpieces.append(new_wp)

        return new_workpieces

    def regenerate_from_sketch(self) -> None:
        """
        Regenerates the workpiece from its sketch definition.

        This method:
        1. Fetches the Sketch definition using self.sketch_uid
        2. Solves a CLONE of the sketch (to match boundaries behavior)
        3. Calculates the natural size from the solved geometry's bounding box
        4. Updates self.natural_width_mm and self.natural_height_mm
        5. Invalidates the geometry cache
        6. Sends an updated signal
        """
        if not self.sketch_uid:
            logger.warning(
                f"WP {self.uid[:8]}: No sketch_uid to regenerate from"
            )
            return

        sketch_def = self.get_sketch_definition()
        if not sketch_def:
            logger.warning(
                f"WP {self.uid[:8]}: Could not find sketch definition "
                f"{self.sketch_uid}"
            )
            return

        logger.debug(
            f"WP {self.uid[:8]}: Regenerating from sketch "
            f"{self.sketch_uid[:8]}"
        )

        # Use a clone to ensure we solve independently of shared state,
        # mirroring the behavior in the `boundaries` property.
        from .sketcher.sketch import Sketch

        instance_sketch = Sketch.from_dict(sketch_def.to_dict())

        # Solve the sketch with current parameter overrides
        success = instance_sketch.solve(variable_overrides=self.sketch_params)
        if not success:
            logger.warning(
                f"WP {self.uid[:8]}: Sketch solve failed during regeneration"
            )
            return

        # Get the solved geometry and calculate its bounding box
        geometry = instance_sketch.to_geometry()
        if geometry.is_empty():
            logger.warning(
                f"WP {self.uid[:8]}: Sketch geometry is empty after solve. "
                "Natural size not updated."
            )
        else:
            # Calculate bounding box in mm
            min_x, min_y, max_x, max_y = geometry.rect()
            width = max(max_x - min_x, 1e-9)  # Prevent zero size
            height = max(max_y - min_y, 1e-9)  # Prevent zero size

            self.natural_width_mm = width
            self.natural_height_mm = height

            logger.debug(
                f"WP {self.uid[:8]}: New natural size: "
                f"{width:.2f}x{height:.2f}mm"
            )

        # Invalidate the geometry cache to force regeneration on next render
        self.clear_render_cache()

        # Send updated signal to trigger UI updates
        self.updated.send(self)
