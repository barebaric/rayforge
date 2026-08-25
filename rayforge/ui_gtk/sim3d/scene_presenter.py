"""
Scene presenter for the 3D canvas.

Owns scene compilation scheduling, the compiled artifact, the playback
OpPlayer, and the playback overlay binding.  Constructed by Canvas3D with
injected callables so it never reaches back into the widget; the canvas
keeps the GL lifecycle and per-frame rendering.
"""

import logging
import math
import time
from collections.abc import Callable
from gettext import gettext as _
from typing import TYPE_CHECKING, Optional

import numpy as np
from blinker import Signal
from raygeo.ops import Ops

from ...context import RayforgeContext
from ...core.optical import material_burn_response
from ...core.workpiece import WorkPiece
from ...machine.kinematic_mapping import (
    KinematicMapping,
    build_layer_assembly,
    resolve_layer_rotary,
)
from ...pipeline.artifact.handle import BaseArtifactHandle
from ...pipeline.artifact.job import JobArtifact
from ...pipeline.artifact.material_state import MaterialStateArtifact
from ...shared.tasker import Task, task_mgr
from ...simulator.op_player import OpPlayer, build_snapshots
from ...simulator.scene3d import (
    CompiledSceneArtifact,
    LayerRenderConfig,
    RenderConfig3D,
    WorkpieceImage,
    compile_scene_from_job,
    compile_stock_scene,
)
from ...simulator.scene3d.cylinder_compiler import generate_cylinder_vertices
from ...simulator.scene3d.stock_compiler import DEFAULT_THICKNESS_MM
from .camera import ViewDirection
from .render_context import SceneVisibility

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor
    from ...machine.assembly import Assembly
    from .renderer.scene_renderer import SceneRenderer
    from .theme_resolver import ThemeResolver
    from .viewport import ViewportConfig

logger = logging.getLogger(__name__)

# Target pixel density for rendered workpiece base images, capped so
# preview textures stay cheap to upload and render.
WORKPIECE_IMAGE_PX_PER_MM = 8.0
MAX_WORKPIECE_IMAGE_DIM = 2048


def _workpiece_image_pixels(wp: WorkPiece) -> np.ndarray | None:
    """Renders a workpiece's base image to RGBA uint8 pixels.

    Runs in a worker thread.  Row 0 is the bottom of the image so the
    texture maps v-up onto the quad, matching the other renderers.
    """
    size = wp.size
    if size[0] <= 1e-9 or size[1] <= 1e-9:
        return None
    width_px = max(1, round(size[0] * WORKPIECE_IMAGE_PX_PER_MM))
    height_px = max(1, round(size[1] * WORKPIECE_IMAGE_PX_PER_MM))
    scale = min(
        1.0,
        MAX_WORKPIECE_IMAGE_DIM / width_px,
        MAX_WORKPIECE_IMAGE_DIM / height_px,
    )
    width_px = max(1, round(width_px * scale))
    height_px = max(1, round(height_px * scale))
    try:
        vips_image = wp.get_vips_image(width_px, height_px)
        if vips_image is None:
            return None
        from ...image.util.vips import normalize_to_rgba

        rgba = normalize_to_rgba(vips_image)
        if rgba is None:
            return None
        memory = rgba.write_to_memory()
        pixels = np.frombuffer(memory, dtype=np.uint8).reshape(
            rgba.height, rgba.width, 4
        )
        return np.ascontiguousarray(pixels)
    except Exception:
        logger.exception("Failed to render workpiece base image %s", wp.name)
        return None


def _workpiece_cylinder_grid(
    world_matrix: np.ndarray,
    diameter: float,
    reverse: bool,
) -> np.ndarray:
    """Grid matrix mapping the image quad onto cylinder-space degrees.

    The image's planar Y (surface millimetres) becomes the rotation
    angle around the cylinder: ``360 / (pi * diameter)`` degrees per
    surface millimetre, matching the rotary Y->degree bake in raygeo.
    The X axis stays along the cylinder.
    """
    sign = -1.0 if reverse else 1.0
    deg_per_mm = sign * 360.0 / (math.pi * diameter)
    deg_scale = np.diag([1.0, deg_per_mm, 1.0, 1.0]).astype(np.float32)
    return deg_scale @ np.asarray(world_matrix, dtype=np.float32)


def _render_workpiece_images(
    workpieces: list[WorkPiece],
    matrices: list[np.ndarray],
    rotary_specs: list[tuple[np.ndarray, float, bool] | None],
) -> list[WorkpieceImage]:
    """Renders workpiece base images and pairs them with their matrices.

    Rotary workpieces (``rotary_specs`` entry is not ``None``) get
    their quad meshed onto the cylinder surface, so the base image
    wraps around the cylinder exactly like the engrave texture.
    """
    images: list[WorkpieceImage] = []
    for wp, matrix, rspec in zip(workpieces, matrices, rotary_specs):
        pixels = _workpiece_image_pixels(wp)
        if pixels is None:
            continue
        cylinder_vertices = None
        rotary_diameter = 0.0
        if rspec is not None:
            world_matrix, diameter, reverse = rspec
            grid_matrix = _workpiece_cylinder_grid(
                world_matrix, diameter, reverse
            )
            cylinder_vertices = generate_cylinder_vertices(
                grid_matrix, diameter
            )
            if cylinder_vertices is not None:
                rotary_diameter = diameter
            else:
                cylinder_vertices = None
        images.append(
            WorkpieceImage(
                pixels=pixels,
                model_matrix=np.asarray(matrix, dtype=np.float32),
                cylinder_vertices=cylinder_vertices,
                rotary_diameter=rotary_diameter,
            )
        )
    return images


class ScenePresenter:
    """
    Compiles the scene, builds the playback player, and binds playback.

    The canvas owns the GL context and per-frame render state; this class
    owns everything that turns a document + job artifact into a compiled
    ``CompiledSceneArtifact`` and an ``OpPlayer``.  Dependencies are
    injected as callables so the presenter stays independent of the widget.
    """

    def __init__(
        self,
        context: RayforgeContext,
        doc_editor: "DocEditor",
        scene: "SceneRenderer",
        *,
        theme_resolver: "ThemeResolver",
        get_viewport: Callable[[], "ViewportConfig"],
        get_gl_initialized: Callable[[], bool],
        get_camera_available: Callable[[], bool],
        make_current: Callable[[], None],
        mark_scene_dirty: Callable[[], None],
        mark_artifact_dirty: Callable[[], None],
        reset_view: Callable[[ViewDirection], None],
        request_render: Callable[[], None],
        upload_complete: Signal,
    ):
        self._context = context
        self._doc_editor = doc_editor
        self._scene = scene
        self._theme_resolver = theme_resolver
        self._get_viewport = get_viewport
        self._get_gl_initialized = get_gl_initialized
        self._get_camera_available = get_camera_available
        self._make_current = make_current
        self._mark_scene_dirty = mark_scene_dirty
        self._mark_artifact_dirty = mark_artifact_dirty
        self._reset_view = reset_view
        self._request_render = request_render
        self._upload_complete = upload_complete

        self.visibility = SceneVisibility()

        self.stock_top_z: float = 0.0
        self.has_z_axis: bool = True

        self._scene_preparation_task: Task | None = None
        self._compiled_artifact: CompiledSceneArtifact | None = None
        self._current_job_handle: BaseArtifactHandle | None = None
        self._compiled_job_generation: int | None = None
        self._op_player: OpPlayer | None = None
        self._playback_assembly: Assembly | None = None
        self._playback_overlay = None
        self._workpiece_image_task: Task | None = None
        self._workpiece_image_generation = 0
        self._workpiece_images: list[WorkpieceImage] = []
        # Folded material state per stock uid (burn surface map data),
        # updated from Pipeline.material_state_ready.
        self._material_states: dict[str, dict] = {}

    def connect(self):
        """Subscribe to the pipeline and upload events that drive the scene.

        Called once the canvas has realized its GL context.  ``connect`` /
        ``disconnect`` pair keeps the presenter's signal wiring in one
        place instead of being threaded through the canvas.
        """
        self._upload_complete.connect(self._on_upload_complete)
        pipeline = self._doc_editor.pipeline
        if pipeline:
            pipeline.processing_state_changed.connect(
                self._on_pipeline_state_changed
            )
            pipeline.job_generation_finished.connect(
                self._on_job_generation_finished
            )
            pipeline.material_state_ready.connect(
                self._on_material_state_ready
            )
        # The pipeline may have already produced material states before
        # this presenter connected (the 3D canvas realizes after the
        # job settles); pick those up so they are not lost.
        if self._refresh_material_states():
            self.update_scene_from_doc()

    def disconnect(self):
        """Unsubscribe from pipeline and upload events."""
        self._upload_complete.disconnect(self._on_upload_complete)
        pipeline = self._doc_editor.pipeline
        if pipeline:
            pipeline.processing_state_changed.disconnect(
                self._on_pipeline_state_changed
            )
            pipeline.job_generation_finished.disconnect(
                self._on_job_generation_finished
            )
            pipeline.material_state_ready.disconnect(
                self._on_material_state_ready
            )

    @property
    def doc(self) -> "Doc":
        """Returns the current document from the editor."""
        return self._doc_editor.doc

    @property
    def op_player(self) -> OpPlayer | None:
        """The current playback player, or None."""
        return self._op_player

    @property
    def playback_assembly(self) -> Optional["Assembly"]:
        """The throwaway assembly for the current playback layer, or None."""
        return self._playback_assembly

    @property
    def compiled_artifact(self) -> CompiledSceneArtifact | None:
        """The last compiled scene artifact, or None."""
        return self._compiled_artifact

    @property
    def scene_preparation_task(self) -> Task | None:
        """The in-flight scene compilation task, or None."""
        return self._scene_preparation_task

    @property
    def job_handle(self) -> BaseArtifactHandle | None:
        """The job artifact handle driving the scene, or None."""
        return self._current_job_handle

    @job_handle.setter
    def job_handle(self, handle: BaseArtifactHandle | None):
        self._current_job_handle = handle

    @property
    def playback_overlay(self):
        """The attached playback overlay widget, or None."""
        return self._playback_overlay

    @property
    def workpiece_images(self) -> list[WorkpieceImage]:
        """The last workpiece base-image scene items, used for picking."""
        return self._workpiece_images

    def set_playback_overlay(self, overlay):
        """Store the playback overlay so players can be bound to it."""
        self._playback_overlay = overlay

    def set_show_travel_moves(self, visible: bool) -> None:
        """Sets travel-move visibility.

        Travel geometry is excluded at upload time, so toggling it needs
        a renderer rebuild rather than a cheap re-render.
        """
        if self.visibility.show_travel_moves == visible:
            return
        self.visibility.show_travel_moves = visible
        self.update_renderers_from_artifact()

    def set_show_grid(self, visible: bool) -> None:
        """Sets grid visibility and requests a re-render."""
        if self.visibility.show_grid == visible:
            return
        self.visibility.show_grid = visible
        self._request_render()

    def set_show_nogo_zones(self, visible: bool) -> None:
        """Sets no-go zone visibility and requests a re-render."""
        if self.visibility.show_nogo_zones == visible:
            return
        self.visibility.show_nogo_zones = visible
        self._request_render()

    def set_show_models(self, visible: bool) -> None:
        """Sets machine model visibility and requests a re-render."""
        if self.visibility.show_models == visible:
            return
        self.visibility.show_models = visible
        self._request_render()

    def set_show_stock(self, visible: bool) -> None:
        """Sets stock visibility and requests a re-render."""
        if self.visibility.show_stock == visible:
            return
        self.visibility.show_stock = visible
        self._request_render()

    def set_show_workpiece_image(self, visible: bool) -> None:
        """Sets workpiece image visibility and requests a re-render."""
        if self.visibility.show_workpiece_image == visible:
            return
        self.visibility.show_workpiece_image = visible
        self._request_render()

    def cancel_scene_preparation(self):
        """Cancel any in-flight scene compilation task."""
        if self._scene_preparation_task:
            self._scene_preparation_task.cancel()
            self._scene_preparation_task = None
        if self._workpiece_image_task:
            self._workpiece_image_task.cancel()
            self._workpiece_image_task = None

    def has_stale_job(self) -> bool:
        """True if the cached job handle is from an older generation."""
        handle = self._current_job_handle
        if handle is None:
            return True
        return (
            handle.generation_id
            != self._doc_editor.pipeline.data_generation_id
        )

    def _on_pipeline_state_changed(self, sender, *, is_processing: bool):
        """
        Handler for when the pipeline's busy state changes. When it becomes
        not busy, the document has settled and the scene should be updated.
        """
        if not is_processing and self._current_job_handle is not None:
            if self.has_stale_job():
                logger.debug(
                    "Pipeline settled with stale job. Clearing 3D scene."
                )
                self._current_job_handle = None
                self._compiled_job_generation = None
                self._compiled_artifact = None
                self._mark_artifact_dirty()
                self._request_render()
            else:
                if (
                    self._current_job_handle.generation_id
                    == self._compiled_job_generation
                ):
                    logger.debug(
                        "Scene already compiled for this "
                        "generation; skipping duplicate update."
                    )
                else:
                    logger.debug("Pipeline has settled. Updating 3D scene.")
                    self.update_scene_from_doc()

    def _on_job_generation_finished(self, sender, **kwargs):
        task_status = kwargs.get("task_status")
        handle = kwargs.get("handle")
        logger.debug(
            f"_on_job_generation_finished: "
            f"status={task_status}, handle={'yes' if handle else 'none'}"
        )
        if task_status == "completed":
            if handle is not None:
                self._current_job_handle = handle
                self.update_scene_from_doc()
                self._request_render()
            else:
                logger.debug("Job completed with no output. Clearing scene.")
                self._current_job_handle = None
                self._compiled_job_generation = None
                self._compiled_artifact = None
                self._mark_artifact_dirty()
                self._request_render()

    def _on_upload_complete(self, sender=None, **_kwargs):
        self._build_op_player_async()
        if self._compiled_artifact and self._op_player:
            self._scene.extract_playback_offsets(self._compiled_artifact)

    @staticmethod
    def _add_material_optical(spec: dict, material) -> None:
        """Stamp the material's resolved optical fields onto a stock spec.

        Adds the absorption dict and burn-response char curve parameters
        so the stock compiler/renderer can wire them into the shader's
        physical burn block. The wavelength is read from the burn entry
        at compile time (it travels with the surface map); here we only
        resolve the material-side response.
        """
        appearance = getattr(material, "appearance", None)
        absorption = None
        if appearance is not None:
            absorption = appearance.extra.get("absorption")
        if isinstance(absorption, dict):
            spec["absorption"] = absorption
        spec["burn_response"] = material_burn_response(material)

    def _store_material_state(
        self, stock_uid: str, stock_name: str, handle
    ) -> bool:
        """Resolve a material-state artifact handle into the burn dict.

        Returns ``True`` if ``_material_states`` changed (i.e. a scene
        update should be triggered), ``False`` otherwise (duplicate or
        no change).
        """
        try:
            artifact = self._context.artifact_store.get(handle)
        except RuntimeError:
            logger.exception("Failed to resolve material state artifact")
            return False
        if not isinstance(artifact, MaterialStateArtifact):
            return False
        state = artifact.material_state
        if state.surface_map is not None and state.grid is not None:
            grid = state.grid
            burn = {
                "handle_key": handle.key,
                "surface_map": state.surface_map,
                "origin_mm": tuple(grid.origin_mm),
                "px_per_mm": tuple(grid.px_per_mm),
                "size_px": tuple(grid.size_px),
                "wavelength_nm": float(state.wavelength_nm),
                "max_power_watts": float(state.max_power_watts),
            }
            previous = self._material_states.get(stock_uid)
            if previous is not None and previous["handle_key"] == handle.key:
                return False
            self._material_states[stock_uid] = burn
            logger.info(
                "Burn surface map ready for stock %r (grid %sx%s px)",
                stock_name,
                grid.size_px[0],
                grid.size_px[1],
            )
            return True
        if stock_uid in self._material_states:
            self._material_states.pop(stock_uid, None)
            return True
        return False

    def _refresh_material_states(self) -> bool:
        """Pick up any material states the Pipeline already computed.

        The pipeline can finish (and deliver a fold's material state)
        before this presenter connects (the 3D canvas realizes after
        the job settles). Relying on the live signal alone would miss
        that state, so re-read the Pipeline's stored handles and drop
        any that are no longer present. Returns whether anything
        changed.
        """
        pipeline = self._doc_editor.pipeline
        if pipeline is None:
            return False
        handles = getattr(pipeline, "_material_state_handles", {}) or {}
        changed = False
        seen: set[str] = set()
        hosts: list[tuple[str, str]] = [
            (item.uid, item.name) for item in self.doc.stock_items
        ]
        hosts.extend(
            (layer.uid, layer.name)
            for layer in self.doc.layers
            if layer.rotary_enabled
        )
        for uid, name in hosts:
            if uid not in handles:
                continue
            seen.add(uid)
            if self._store_material_state(uid, name, handles[uid]):
                changed = True
        for uid in list(self._material_states):
            if uid not in seen:
                self._material_states.pop(uid, None)
                changed = True
        return changed

    def _on_material_state_ready(self, sender, **kwargs):
        """Store a material host's folded burn surface map and recompile.

        The fold emits during the pipeline run; the stock specs pick
        the burn data up on the next scene compile, which reuses the
        current job handle when the pipeline is still running.
        """
        item = kwargs.get("item")
        handle = kwargs.get("handle")
        if item is None or handle is None:
            return
        if self._store_material_state(item.uid, item.name, handle):
            self.update_scene_from_doc()

    def _build_op_player_async(self):
        ops = self._get_ops_for_playback()
        time_ops = self._get_time_ops_for_playback()
        machine = self._context.machine
        if machine is None:
            return

        if ops is None or ops.is_empty():
            self._op_player = None
            self._playback_assembly = None
            for renderer in self._scene.ops_renderers:
                renderer.powered_offsets = np.array([], dtype=np.int32)
                renderer.travel_offsets = np.array([], dtype=np.int32)
            for renderer in self._scene.ring_renderers:
                renderer.ring_offsets = np.array([], dtype=np.int32)
            if self._playback_overlay:
                self._playback_overlay.set_player(None)
            self._request_render()
            return

        # Preserve the playhead and seek snapshots when the underlying
        # ops object has not changed (e.g. only the viewport moved).
        saved_index = None
        reused_snapshots = []
        if self._op_player is not None and self._op_player.ops is ops:
            saved_index = self._op_player.current_index
            reused_snapshots = self._op_player.snapshots

        player = OpPlayer(
            ops,
            machine,
            self.doc,
            build_snapshots=False,
            time_ops=time_ops,
        )
        player.set_playback_params(
            machine.max_cut_speed,
            machine.max_travel_speed,
            machine.acceleration,
        )
        player.set_snapshots(reused_snapshots)

        # Make the player available right away so that the next render
        # can dim textures that have not been reached yet.  Seeking to
        # the first layer is cheap (the first LAYER_START is near the
        # start of the ops), and reused snapshots keep restores of a
        # previous playhead fast as well.
        if saved_index is not None:
            player.seek(saved_index)
            initial_index = saved_index
        else:
            player.seek_to_first_layer()
            initial_index = 0
        self._op_player = player
        player.layer_changed.connect(self._on_playback_layer_changed)
        self._on_playback_layer_changed(player)
        if self._playback_overlay:
            self._playback_overlay.set_player(player, initial_index)
        self._request_render()

        # Build seek-acceleration snapshots in the background.  They are
        # collected into a fresh list and attached from the main thread
        # to avoid racing with concurrent seeks reading _snapshots.
        def _on_snapshots_done(task):
            if task.get_status() != "completed":
                return
            if self._op_player is player:
                player.set_snapshots(task.result())

        task_mgr.run_thread(
            build_snapshots,
            ops,
            machine,
            self.doc,
            key=(id(self), "build-snapshots"),
            when_done=_on_snapshots_done,
        )

    def _on_playback_layer_changed(self, player, layer_uid=None, **_kwargs):
        """Rebuild the throwaway playback assembly for the current layer.

        Connected to ``OpPlayer.layer_changed`` and also called once on
        player creation.  Resolves the effective layer (current or the
        first layer while in the preamble) and updates the scene's
        cylinder transform without mutating the live machine.
        """
        machine = self._context.machine
        if machine is None or player is None:
            return
        layer = player.get_effective_layer(self.doc)
        assembly = build_layer_assembly(machine, layer)
        self._playback_assembly = assembly
        if assembly.has_rotary:
            self._scene.set_cylinder_transform(
                assembly.cylinder_base_transform()
            )
        else:
            self._scene.set_cylinder_transform(np.eye(4, dtype=np.float64))
        self._request_render()

    def _on_scene_prepared(self, task: Task):
        """
        Callback for when the background scene compilation task is
        finished.  The compiled artifact is available directly as
        ``task.result_value`` since the compilation runs in-process.
        """
        if task.get_status() != "completed":
            if task.is_cancelled():
                logger.debug("Scene preparation task cancelled (superseded).")
            else:
                self._compiled_artifact = None
                self._op_player = None
                self._playback_assembly = None
                logger.error("Scene preparation task failed.")
                self._mark_artifact_dirty()
                self._request_render()
            return

        self._scene_preparation_task = None

        artifact = task.result()
        if artifact is None:
            logger.warning(
                "Scene task completed but produced no "
                "artifact (possibly empty scene)."
            )
            self._compiled_artifact = None
            self._mark_artifact_dirty()
            self._request_render()
            return

        if not isinstance(artifact, CompiledSceneArtifact):
            logger.error(
                f"Expected CompiledSceneArtifact, got "
                f"{type(artifact).__name__}"
            )
            self._compiled_artifact = None
            self._mark_artifact_dirty()
            self._request_render()
            return

        logger.debug("Scene compilation finished.")
        self._compiled_artifact = artifact
        self._mark_artifact_dirty()
        self._request_render()

    def update_renderers_from_artifact(self):
        if not self._compiled_artifact:
            for renderer in self._scene.ops_renderers:
                renderer.clear()
            for renderer in self._scene.ring_renderers:
                renderer.clear()
                renderer.ring_offsets = np.array([], dtype=np.int32)
            self._request_render()
            return

        if not self._get_gl_initialized():
            return

        self._make_current()

        self._scene.update_from_artifact(
            self._compiled_artifact, self.visibility.show_travel_moves
        )

        self._theme_resolver.update_renderer_color_luts()

        logger.debug(
            "Scanline overlay uploaded. Groups: {}".format(
                ", ".join(
                    "{}:{}".format(
                        "rot" if r.is_rotary else "flat",
                        r.vertex_count,
                    )
                    for r in self._scene.ring_renderers
                )
            )
        )

        self._request_render()

    def _get_ops_for_playback(self) -> Ops | None:
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact):
                return artifact.preview_ops
        return None

    def _get_time_ops_for_playback(self) -> Ops | None:
        """Unmapped ops for the playback time model.

        The preview ops of rotary jobs keep endpoint Y at a constant
        while the real rotation lives in extra axes, which distorts
        distances and makes arcs degenerate. The raw assembled ops
        carry the true (unwrapped) path, so durations must come from
        them; command indices and order match the preview ops 1:1.
        """
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact):
                return artifact.ops
        return None

    def _build_stock_specs(
        self, viewport: "ViewportConfig", machine
    ) -> list[dict]:
        """Collect visible stock into plain-data compiler specs.

        Flat specs come from the document's stock items (world-space
        geometry rings, thickness, material parameters).  Rotary specs
        come from rotary layers with a selected stock material and
        carry the layer's object diameter plus the renderable axial
        length.  The heavy meshing runs on the background compile
        thread, so the spec dicts stay serializable.
        """
        specs: list[dict] = []
        for item in self.doc.stock_items:
            if not item.visible:
                continue
            asset = item.stock_asset
            if asset is None or asset.geometry.is_empty():
                continue
            thickness = asset.thickness
            if thickness is not None and thickness <= 0:
                continue
            world_geo = item.get_world_geometry()
            if world_geo.is_empty():
                continue
            outers, holes = world_geo.split_inner_and_outer_polygons()
            if not outers:
                continue

            material = item.material
            if material is None:
                material = self._context.material_mgr.get_default_material()
            appearance = material.appearance
            texture_path = material.get_texture_path()

            spec = {
                "name": item.name,
                "thickness": (
                    float(thickness) if thickness is not None else None
                ),
                "outers": outers,
                "holes": holes,
                "texture_path": (
                    str(texture_path) if texture_path is not None else None
                ),
                "texture_size_mm": float(appearance.texture_size_mm),
                "roughness": float(appearance.roughness),
                "metallic": float(appearance.metallic),
                "color": appearance.color,
                "tint": item.get_effective_color(),
            }
            self._add_material_optical(spec, material)
            burn = self._material_states.get(item.uid)
            if burn is not None:
                spec["burn"] = dict(burn)
            specs.append(spec)

        if machine is not None:
            specs.extend(self._build_rotary_stock_specs(viewport, machine))
        return specs

    def _build_rotary_stock_specs(
        self, viewport: "ViewportConfig", machine
    ) -> list[dict]:
        """Collect rotary layers into stock specs.

        The axial length matches the work area width capped by the
        default rotary module's maximum workpiece length.
        """
        specs: list[dict] = []
        max_length = viewport.width_mm
        default_rm = machine.get_default_rotary_module()
        if default_rm:
            max_length = min(max_length, default_rm.max_workpiece_length)

        for layer in self.doc.layers:
            if not layer.rotary_enabled:
                continue
            if layer.rotary_diameter <= 0:
                continue
            material = layer.stock_material
            if material is None:
                material = self._context.material_mgr.get_default_material()
            appearance = material.appearance
            texture_path = material.get_texture_path()
            spec = {
                "name": _("{layer} stock").format(layer=layer.name),
                "kind": "rotary",
                "diameter": float(layer.rotary_diameter),
                "length": float(max_length),
                "texture_path": (
                    str(texture_path) if texture_path is not None else None
                ),
                "texture_size_mm": float(appearance.texture_size_mm),
                "roughness": float(appearance.roughness),
                "metallic": float(appearance.metallic),
                "color": appearance.color,
            }
            self._add_material_optical(spec, material)
            burn = self._material_states.get(layer.uid)
            if burn is not None:
                spec["burn"] = dict(burn)
            specs.append(spec)
        return specs

    def update_scene_from_doc(self):
        """
        Updates the entire scene content from the document. This is the main
        entry point for refreshing the 3D view.
        """
        if not self._get_gl_initialized():
            return

        t_update_start = time.perf_counter()
        logger.debug("Canvas3D: Updating scene from document.")

        # Folded material states may arrive after the last update (or
        # before this presenter connected); re-read the Pipeline's
        # stored states so the stock specs always carry the burn.
        self._refresh_material_states()

        # Theme/color updates only need to happen once per theme change
        if self._theme_resolver.theme_is_dirty:
            self._theme_resolver.update_theme_and_colors()
        if not self._theme_resolver.color_set:
            logger.warning("Cannot update scene, color set not resolved.")
            return

        viewport = self._get_viewport()

        # Update cylinder renderers and camera based on layer rotary state
        any_rotary = any(layer.rotary_enabled for layer in self.doc.layers)
        self._mark_scene_dirty()
        if (
            self._scene.had_rotary_layers
            and not any_rotary
            and self._get_camera_available()
        ):
            self._reset_view(ViewDirection.ISO)
        self._scene.had_rotary_layers = any_rotary

        world_to_visual = np.identity(4, dtype=np.float32)
        world_to_cyl_local = np.identity(4, dtype=np.float32)

        machine = self._context.machine
        has_z_axis = True
        if machine:
            has_z_axis = machine.has_z_axis

        stock_specs = self._build_stock_specs(viewport, machine)
        stock_top_z = self._compute_stock_top_z(stock_specs)
        self.stock_top_z = stock_top_z
        self.has_z_axis = has_z_axis

        if machine:
            if has_z_axis:
                # Has-Z machines render content at its authored Z
                # (plus WCS Z), faithful to the coordinates.
                world_to_visual = self._compute_world_to_visual(
                    viewport, machine
                )
            else:
                # No-Z machines author everything at z=0 (the stock
                # top in the toolpath convention), so lift content
                # onto the stock top in visual space.
                wcs = viewport.wcs_offset_mm
                content_z = wcs[2] + stock_top_z
                world_to_visual = self._compute_world_to_visual(
                    viewport, machine, z_offset=content_z
                )

            asm = machine.assembly
            if asm.has_rotary:
                self._scene.set_cylinder_transform(
                    asm.cylinder_base_transform()
                )
            else:
                self._scene.set_cylinder_transform(np.eye(4, dtype=np.float64))
        else:
            world_to_visual = self._compute_world_to_visual(viewport, None)

        # Stock is always bed-anchored (Z=0).
        stock_world_to_visual = self._compute_world_to_visual(
            viewport, machine, z_offset=0.0
        )

        layer_configs: dict[str, LayerRenderConfig] = {}
        for layer in self.doc.layers:
            axis_position = 0.0
            reverse = False
            axis_position_3d = None
            cylinder_dir = None
            if layer.rotary_enabled and machine:
                cfg = resolve_layer_rotary(layer, machine)
                module = cfg.module
                if module is not None:
                    mapping = KinematicMapping.from_rotary_module(
                        module,
                        layer.rotary_diameter,
                        apply_gear_ratio=False,
                    )
                    if mapping is not None:
                        axis_position = mapping.axis_position
                        axis_position_3d = tuple(
                            mapping.axis_position_3d.tolist()
                        )
                        cylinder_dir = tuple(mapping.cylinder_dir.tolist())
                        reverse = mapping.reverse
            layer_configs[layer.uid] = LayerRenderConfig(
                rotary_enabled=layer.rotary_enabled,
                rotary_diameter=layer.rotary_diameter,
                axis_position=axis_position,
                reverse=reverse,
                axis_position_3d=axis_position_3d,
                cylinder_dir=cylinder_dir,
            )

        render_config = RenderConfig3D(
            world_to_visual=world_to_visual,
            world_to_cyl_local=world_to_cyl_local,
            stock_world_to_visual=stock_world_to_visual,
            stock_top_z=stock_top_z,
            has_z_axis=has_z_axis,
            layer_configs=layer_configs,
            stock_specs=stock_specs,
        )

        self._schedule_scene_preparation(render_config.to_dict())
        self.update_workpiece_images_from_doc(world_to_visual)

        t_update_elapsed = (time.perf_counter() - t_update_start) * 1000
        if t_update_elapsed > 5:
            logger.debug(
                f"update_scene_from_doc took {t_update_elapsed:.1f}ms"
            )

    def _schedule_scene_preparation(
        self,
        render_config_dict: dict,
    ):
        if (
            not self._get_gl_initialized()
            or self._theme_resolver.color_set is None
        ):
            return

        job_handle = self._current_job_handle
        if job_handle is None and not render_config_dict.get("stock_specs"):
            # Nothing to compile at all; drop any previously compiled
            # content so a deleted stock does not linger.
            logger.debug("No job artifact and no stock, skipping compilation.")
            if self._compiled_artifact is not None:
                self._compiled_artifact = None
                self._mark_artifact_dirty()
                self._request_render()
            return

        if self._scene_preparation_task:
            self._scene_preparation_task.cancel()
            self._scene_preparation_task = None
            logger.debug(
                "Cancelled in-progress compilation, scheduling new one."
            )

        if job_handle is None:
            # Stock is document content and must render even without
            # an assembled job; compile a stock-only scene.
            logger.debug("Scheduling stock-only compilation.")
            self._run_scene_preparation_task(
                compile_stock_scene,
                render_config_dict,
            )
            return

        logger.debug("Scheduling scene compilation task.")
        self._compiled_job_generation = job_handle.generation_id
        self._run_scene_preparation_task(
            compile_scene_from_job,
            self._context.artifact_store,
            job_handle.to_dict(),
            render_config_dict,
        )

    def _run_scene_preparation_task(self, fn, *args):
        """Runs a scene compilation function in the worker thread."""
        task_key = (id(self), "prepare-3d-scene-vertices")
        self._scene_preparation_task = task_mgr.run_thread(
            fn,
            *args,
            key=task_key,
            when_done=self._on_scene_prepared,
        )

    @staticmethod
    def _compute_world_to_visual(
        viewport: "ViewportConfig", machine, z_offset: float | None = None
    ) -> np.ndarray:
        """Builds the world->visual matrix from the viewport and machine.

        ``z_offset`` overrides the WCS Z translation.  When *None* the
        WCS Z from the viewport is used (the original behaviour).
        """
        world_to_visual = np.identity(4, dtype=np.float32)
        ms = viewport.margin_shift
        if z_offset is None:
            wcs = viewport.wcs_offset_mm
            z = wcs[2]
        else:
            z = z_offset
        world_to_visual[0, 3] = ms[0, 3]
        world_to_visual[1, 3] = ms[1, 3]
        world_to_visual[2, 3] = z
        return world_to_visual

    @staticmethod
    def _compute_stock_top_z(
        stock_specs: list[dict],
    ) -> float:
        """Max visible flat-stock thickness (0 if none).

        Rotary stocks are excluded — rotary uses cylinder kinematics,
        not the flat lift.  Uses the same default-thickness fallback as
        :func:`compile_stock_layers` so the lift matches the rendered
        mesh even when the asset has no explicit thickness.
        """
        max_z = 0.0
        for spec in stock_specs:
            if spec.get("kind") == "rotary":
                continue
            thickness = spec.get("thickness")
            if thickness is None:
                t = DEFAULT_THICKNESS_MM
            else:
                try:
                    t = float(thickness)
                except (TypeError, ValueError):
                    t = DEFAULT_THICKNESS_MM
            if t <= 0:
                continue
            max_z = max(max_z, t)
        return max_z

    def update_workpiece_images_from_doc(
        self, world_to_visual: np.ndarray | None = None
    ) -> None:
        """Renders workpiece base images and uploads them to the scene.

        Collects every visible workpiece with a source image, renders
        its base image off the main thread, and feeds the resulting
        textures to the workpiece image renderer.
        """
        if not self._get_gl_initialized():
            return
        renderer = self._scene.workpiece_image_renderer
        if renderer is None:
            return

        if world_to_visual is None:
            machine = self._context.machine
            viewport = self._get_viewport()
            world_to_visual = self._compute_world_to_visual(viewport, machine)

        workpieces: list[WorkPiece] = []
        matrices: list[np.ndarray] = []
        rotary_specs: list[tuple[np.ndarray, float, bool] | None] = []
        machine = self._context.machine
        for wp in self.doc.get_descendants(WorkPiece):
            if wp.source_segment is None:
                continue
            if wp.geometry_provider_uid:
                provider = self.doc.get_asset_by_uid(wp.geometry_provider_uid)
                if provider and getattr(provider, "hidden", False):
                    continue
            try:
                world_matrix = wp.get_world_transform().to_4x4_numpy()
            except (ValueError, TypeError, np.linalg.LinAlgError):
                logger.warning(
                    "Skipping workpiece %s: non-invertible transform.",
                    wp.name,
                )
                continue
            workpieces.append(wp)
            matrices.append(world_to_visual @ world_matrix)
            rotary_specs.append(
                self._resolve_workpiece_rotary(wp, machine, world_matrix)
            )

        if not workpieces:
            self._workpiece_images = []
            if renderer.images:
                self._make_current()
                renderer.clear()
                self._request_render()
            return

        if self._workpiece_image_task:
            self._workpiece_image_task.cancel()
            self._workpiece_image_task = None

        self._workpiece_image_generation += 1
        generation = self._workpiece_image_generation
        self._workpiece_image_task = task_mgr.run_thread(
            _render_workpiece_images,
            workpieces,
            matrices,
            rotary_specs,
            key=(id(self), "workpiece-images"),
            when_done=lambda task, g=generation: (
                self._on_workpiece_images_ready(g, task)
            ),
        )

    def _resolve_workpiece_rotary(
        self,
        wp: WorkPiece,
        machine,
        world_matrix: np.ndarray,
    ) -> tuple[np.ndarray, float, bool] | None:
        """Rotary wrap spec for a workpiece image, or ``None`` when flat.

        Returns the world matrix plus the resolved diameter and axis
        direction (``reverse``) of the workpiece's own layer, mirroring
        the per-layer rotary configuration used by the scene compiler.
        """
        if machine is None:
            return None
        layer = wp.layer
        if layer is None or not layer.rotary_enabled:
            return None
        if layer.rotary_diameter is None or layer.rotary_diameter <= 0:
            return None
        reverse = False
        cfg = resolve_layer_rotary(layer, machine)
        if cfg.module is not None:
            mapping = KinematicMapping.from_rotary_module(
                cfg.module,
                layer.rotary_diameter,
                apply_gear_ratio=False,
            )
            if mapping is not None:
                reverse = mapping.reverse
        return (
            np.asarray(world_matrix, dtype=np.float32),
            float(layer.rotary_diameter),
            reverse,
        )

    def _on_workpiece_images_ready(self, generation: int, task: Task) -> None:
        """Uploads freshly rendered workpiece images to the renderer."""
        self._workpiece_image_task = None
        if generation != self._workpiece_image_generation:
            return
        if task.get_status() != "completed":
            return
        renderer = self._scene.workpiece_image_renderer
        if renderer is None or not self._get_gl_initialized():
            return
        try:
            self._make_current()
            images = task.result()
            self._workpiece_images = images
            renderer.set_images(images)
        except Exception:
            logger.exception("Failed to upload workpiece images")
        self._request_render()
