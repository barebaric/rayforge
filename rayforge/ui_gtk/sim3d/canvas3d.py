import logging
import math
import time
from typing import TYPE_CHECKING

from gi.repository import Gdk, Gtk, Pango
from OpenGL import GL
from OpenGL.error import GLError

from ...context import RayforgeContext
from ...pipeline.pipeline import Pipeline
from ...shared.units.formatter import (
    get_default_grid_step_mm,
    get_preferred_unit_factor,
)
from .camera import ViewDirection
from .camera_controller import CameraController
from .chunked_upload import ChunkedUploadController
from .doc_signals import DocSignalHub
from .render_context import FrameInputs, RenderContext
from .renderer.scene_renderer import SceneRenderer
from .scene_presenter import ScenePresenter
from .theme_resolver import ThemeResolver
from .viewport import ViewportConfig

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Canvas3D(Gtk.GLArea):
    """A GTK Widget for rendering a 3D scene with OpenGL."""

    def __init__(
        self,
        context: RayforgeContext,
        doc_editor: "DocEditor",
        viewport: ViewportConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Render only on demand (queue_render), not continuously: GTK 4.22
        # defaults auto_render to True, which renders every frame clock
        # tick and starves lower-priority idle callbacks (task when_done
        # handlers) while wasting GPU work.
        self.set_auto_render(False)
        self._context = context
        self._doc_editor = doc_editor
        self._viewport = viewport

        self._scene = SceneRenderer()
        self._ctx = RenderContext()
        self._show_travel_moves = False
        self._show_nogo_zones = True
        self._show_models = True
        self._show_grid = True
        self._show_ops_underlay = True
        self._gl_initialized = False
        self._scene_gl_dirty = False

        self._theme_resolver = ThemeResolver(
            self,
            scene=self._scene,
            get_machine=lambda: self._context.machine,
            get_gl_initialized=lambda: self._gl_initialized,
            request_render=self.queue_render,
        )

        self._upload_ctrl = ChunkedUploadController(
            self._scene,
            get_artifact=lambda: self._presenter.compiled_artifact,
            get_show_travel_moves=lambda: self._show_travel_moves,
            get_gl_initialized=lambda: self._gl_initialized,
            make_current=self.make_current,
            request_render=self.queue_render,
            on_luts_required=self._theme_resolver.update_renderer_color_luts,
        )

        self._presenter = ScenePresenter(
            self._context,
            self._doc_editor,
            self._scene,
            theme_resolver=self._theme_resolver,
            get_viewport=self._get_viewport,
            get_gl_initialized=lambda: self._gl_initialized,
            get_show_travel_moves=lambda: self._show_travel_moves,
            get_camera_available=lambda: self._cam_ctrl.camera is not None,
            make_current=self.make_current,
            mark_scene_dirty=self._mark_scene_dirty,
            mark_artifact_dirty=lambda: (
                self._upload_ctrl.mark_artifact_dirty()
            ),
            reset_view=self.reset_view,
            request_render=self.queue_render,
            upload_complete=self._upload_ctrl.upload_complete,
        )

        self._cam_ctrl = CameraController(
            self,
            get_viewport=self._get_viewport,
            request_render=self.queue_render,
            on_key_pressed=self._on_key_pressed,
        )

        self._doc_hub = DocSignalHub(
            self._context,
            self._doc_editor,
            set_viewport=self._set_viewport,
            mark_scene_dirty=self._mark_scene_dirty,
            request_render=self.queue_render,
            refresh_scene=self._presenter.update_scene_from_doc,
            get_gl_initialized=lambda: self._gl_initialized,
        )

        self.set_has_depth_buffer(True)
        self.set_focusable(True)
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self._cam_ctrl.on_resize)
        self.connect("notify::style", self._theme_resolver.on_style_changed)

        self._doc_hub.connect()

        self._context.config.changed.connect(self._on_config_changed)

    def set_machine(self, viewport: ViewportConfig | None = None):
        self._doc_hub.set_machine(viewport)

    def has_stale_job(self) -> bool:
        """True if the cached job handle is from an older generation."""
        return self._presenter.has_stale_job()

    @property
    def doc(self) -> "Doc":
        """Returns the current document from the editor."""
        return self._doc_hub.doc

    @property
    def pipeline(self) -> "Pipeline":
        """Returns the current pipeline from the editor."""
        return self._doc_hub.pipeline

    def reset_view(self, direction: ViewDirection):
        """Resets the camera to the specified preset view."""
        self._cam_ctrl.reset_view(direction)

    def set_perspective(self, enabled: bool) -> bool:
        """Toggles the 3D camera between perspective and orthographic.

        Returns True if the camera was available and updated.
        """
        camera = self._cam_ctrl.camera
        if camera is None:
            return False
        camera.is_perspective = enabled
        self.queue_render()
        return True

    def set_playback_overlay(self, overlay):
        """Attach the playback overlay widget and bind it to this canvas."""
        self._presenter.set_playback_overlay(overlay)
        overlay.set_canvas(self)

    def _on_config_changed(self, sender, **kwargs):
        """Updates renderer color LUTs when config settings change."""
        if not self._gl_initialized:
            return
        axis_renderer = self._scene.axis_renderer
        if axis_renderer:
            new_step = get_default_grid_step_mm()
            if not math.isclose(axis_renderer.grid_size_mm, new_step):
                self.make_current()
                axis_renderer.set_grid_size(new_step)
            axis_renderer.set_grid_unit_factor(
                get_preferred_unit_factor("length")
            )
        self._theme_resolver.update_renderer_color_luts()
        self.queue_render()

    def on_realize(self, area) -> None:
        """Called when the GLArea is ready to have its context made current."""
        logger.info("GLArea realized.")

        self._cam_ctrl.create_camera(self.get_width(), self.get_height())

        self._init_gl_resources()
        self._theme_resolver.mark_dirty()

        self.reset_view(ViewDirection.ISO)
        self._theme_resolver.update_theme_and_colors()
        self._presenter.connect()

        if self._presenter.job_handle is None and self.pipeline:
            self._presenter.job_handle = self.pipeline.last_completed_handle

        self._presenter.update_scene_from_doc()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        overlay = self._presenter.playback_overlay
        if keyval == Gdk.KEY_space and overlay:
            overlay.handle_space()
            return True
        return False

    def on_unrealize(self, area) -> None:
        """Called before the GLArea is unrealized."""
        logger.info("GLArea unrealized. Cleaning up GL resources.")
        self._presenter.disconnect()
        self._doc_hub.disconnect()
        self._context.config.changed.disconnect(self._on_config_changed)
        self._upload_ctrl.cancel()
        try:
            self.make_current()
            self._presenter.cancel_scene_preparation()
            self._scene.cleanup()
        except GLError as e:
            logger.debug("Error during GL cleanup on unrealize: %s", e)
        finally:
            self._gl_initialized = False
        logger.debug("on_unrealize: finished.")

    def _init_gl_resources(self) -> None:
        """Initializes OpenGL state, shaders, and renderer objects."""
        try:
            self.make_current()
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            # Get the theme's default font family from GTK
            font_family = "sans-serif"  # A safe fallback
            settings = Gtk.Settings.get_default()
            if settings:
                font_name_str = settings.get_property("gtk-font-name")
                logger.debug(f"Gtk uses font {font_name_str}")
                if font_name_str:
                    # Use Pango to reliably parse the string
                    # (e.g., "Ubuntu Sans")
                    font_desc = Pango.FontDescription.from_string(
                        font_name_str
                    )
                    font_family = font_desc.get_family() or "sans-serif"
                    logger.debug(f"Pango normalized font to {font_family}")

            self._scene.set_viewport(self._viewport)
            self._scene.set_font_family(font_family)
            self._scene.init_gl()

            self._gl_initialized = True
        except Exception:
            logger.exception("OpenGL Initialization Error")
            self._gl_initialized = False

    def _get_viewport(self) -> ViewportConfig:
        """Returns the current viewport configuration."""
        return self._viewport

    def _set_viewport(self, viewport: ViewportConfig):
        """Sets the viewport configuration (from the signal hub)."""
        self._viewport = viewport

    def _mark_scene_dirty(self):
        """Mark the GL scene as needing a rebuild on the next frame."""
        self._scene_gl_dirty = True

    def _process_pending_gl_updates(self):
        if self._scene_gl_dirty:
            self._scene_gl_dirty = False
            if self._scene.update_axis_from_viewport(self._viewport):
                self._theme_resolver.mark_dirty()
            self.make_current()
            self._scene.update_cylinders_from_doc(
                self.doc, self._viewport, self._context.machine
            )
            self._scene.update_models_from_context(
                self._context, self._context.machine
            )
            machine = self._context.machine
            if self._scene.zone_renderer and machine:
                self._scene.update_zones_from_machine(machine)
        self._upload_ctrl.process_pending()

    def on_render(self, area, ctx) -> bool:
        """The main rendering loop."""
        if not self._cam_ctrl.camera or not self._gl_initialized:
            return False

        self._process_pending_gl_updates()

        if self._theme_resolver.theme_is_dirty:
            self._theme_resolver.update_theme_and_colors()

        if not self._theme_resolver.color_set:
            return False

        t_render_start = time.perf_counter()
        try:
            GL.glViewport(
                0, 0, self._cam_ctrl.camera.width, self._cam_ctrl.camera.height
            )
            GL.glClear(
                GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT  # type: ignore
            )

            frame = FrameInputs(
                camera=self._cam_ctrl.camera,
                viewport=self._viewport,
                color_set=self._theme_resolver.color_set,
                op_player=self._presenter.op_player,
                machine=self._context.machine,
                playback_assembly=self._presenter.playback_assembly,
                compiled_artifact=self._presenter.compiled_artifact,
                doc=self.doc,
                cylinder_transform=self._scene.cylinder_transform,
                had_rotary_layers=self._scene.had_rotary_layers,
                show_travel_moves=self._show_travel_moves,
                show_grid=self._show_grid,
                show_nogo_zones=self._show_nogo_zones,
                show_models=self._show_models,
                show_ops_underlay=self._show_ops_underlay,
            )
            self._ctx.update(frame)
            self._scene.prepare(self._ctx)
            self._scene.render(self._ctx, None)

        except Exception:
            logger.exception("OpenGL Render Error")
            return False

        t_render_elapsed = (time.perf_counter() - t_render_start) * 1000
        if t_render_elapsed > 16:
            logger.debug(f"on_render took {t_render_elapsed:.1f}ms")
        return True

    def set_show_travel_moves(self, visible: bool):
        """Sets the visibility of travel moves in the 3D view."""
        if self._show_travel_moves == visible:
            return
        self._show_travel_moves = visible
        self._presenter.update_renderers_from_artifact()

    def set_show_nogo_zones(self, visible: bool):
        if self._show_nogo_zones == visible:
            return
        self._show_nogo_zones = visible
        self.queue_render()

    def set_show_models(self, visible: bool):
        if self._show_models == visible:
            return
        self._show_models = visible
        self.queue_render()

    def set_show_grid(self, visible: bool):
        if self._show_grid == visible:
            return
        self._show_grid = visible
        self.queue_render()

    def set_show_ops_underlay(self, visible: bool):
        if self._show_ops_underlay == visible:
            return
        self._show_ops_underlay = visible
        self.queue_render()

    def update_scene_from_doc(self):
        """Refreshes the 3D scene content from the document."""
        self._presenter.update_scene_from_doc()

    def scene_is_ready(self) -> bool:
        """True when the compiled scene is uploaded and rendered.

        Tooling (e.g. the screenshot harness) polls this to wait for the
        3D view to reach a stable, rendered state.
        """
        if not self._gl_initialized:
            return False
        if self._presenter.compiled_artifact is None:
            return False
        if self._upload_ctrl.is_dirty:
            return False
        task = self._presenter.scene_preparation_task
        return task is None or task.is_final()
