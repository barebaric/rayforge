"""
Base class for OpenGL renderers that manage their own GPU resources.
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, final

from OpenGL import GL

from ..shader.base import Shader

if TYPE_CHECKING:
    from ..gl_utils import ShaderSet
    from ..render_context import RenderContext

logger = logging.getLogger(__name__)


class BaseRenderer(ABC):
    """A base class for an OpenGL renderer that manages its own
    resources."""

    @abstractmethod
    def prepare(self, ctx: "RenderContext") -> None:
        """Per-frame state setup before the draw."""
        raise NotImplementedError

    @abstractmethod
    def render(
        self, ctx: "RenderContext", shaders: "ShaderSet", **kwargs
    ) -> None:
        """Performs the GL draw using the given shaders."""
        raise NotImplementedError

    @abstractmethod
    def init_gl(self) -> None:
        """Creates the renderer's OpenGL resources."""
        raise NotImplementedError

    def __init__(self):
        """Initializes the resource tracking lists."""
        self.shader: Shader | None = None
        self._owned_vaos: list[int] = []
        self._owned_vbos: list[int] = []
        self._owned_textures: list[int] = []
        self._owned_renderers: list[BaseRenderer] = []

    def _create_vao(self) -> int:
        """Creates a VAO and registers it for automatic cleanup."""
        vao = GL.glGenVertexArrays(1)
        self._owned_vaos.append(vao)
        return vao

    def _create_vbo(self) -> int:
        """Creates a VBO and registers it for automatic cleanup."""
        vbo = GL.glGenBuffers(1)
        self._owned_vbos.append(vbo)
        return vbo

    def _create_texture(self) -> int:
        """Creates a Texture and registers it for automatic cleanup."""
        texture = GL.glGenTextures(1)
        self._owned_textures.append(texture)
        return texture

    def _add_child_renderer(self, renderer: "BaseRenderer"):
        """Adds a child renderer to be cleaned up automatically."""
        self._owned_renderers.append(renderer)

    def _remove_child_renderer(self, renderer: "BaseRenderer") -> None:
        """Removes a child renderer from automatic cleanup."""
        try:
            self._owned_renderers.remove(renderer)
        except ValueError:
            pass

    def _delete_owned(self, vao: int = 0, vbo: int = 0) -> None:
        """Deletes owned GL resources and untracks them from cleanup."""
        if vao:
            try:
                self._owned_vaos.remove(vao)
            except ValueError:
                pass
            GL.glDeleteVertexArrays(1, [vao])
        if vbo:
            try:
                self._owned_vbos.remove(vbo)
            except ValueError:
                pass
            GL.glDeleteBuffers(1, [vbo])

    def _cleanup_self(self) -> None:
        """
        A method for subclasses to override for their specific cleanup
        logic.
        """

    @final
    def cleanup(self) -> None:
        """Cleans up all tracked OpenGL resources."""
        try:
            self._cleanup_self()

            for renderer in self._owned_renderers:
                renderer.cleanup()

            if self.shader:
                self.shader.cleanup()

            if self._owned_textures:
                GL.glDeleteTextures(
                    len(self._owned_textures), self._owned_textures
                )
                self._owned_textures.clear()

            if self._owned_vaos:
                GL.glDeleteVertexArrays(
                    len(self._owned_vaos), self._owned_vaos
                )
                self._owned_vaos.clear()
            if self._owned_vbos:
                GL.glDeleteBuffers(len(self._owned_vbos), self._owned_vbos)
                self._owned_vbos.clear()
        except GL.GLError:
            logger.exception("Error during renderer cleanup")
