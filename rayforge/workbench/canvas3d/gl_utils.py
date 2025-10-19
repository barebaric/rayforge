"""
A collection of utility classes and functions for simplifying common
PyOpenGL tasks, such as shader compilation and buffer management.
"""

import logging
from typing import Optional, Union, final

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

logger = logging.getLogger(__name__)


class Shader:
    """Manages a GLSL shader program, including compilation and uniforms."""

    def __init__(self, vertex_source: str, fragment_source: str):
        """
        Compiles and links the vertex and fragment shader sources.

        Args:
            vertex_source: The source code for the vertex shader.
            fragment_source: The source code for the fragment shader.

        Raises:
            Exception: If shader compilation or linking fails.
        """
        # Determine the correct GLSL header for the current context.
        version_str = GL.glGetString(GL.GL_VERSION)
        is_es = version_str is not None and b"OpenGL ES" in version_str
        if is_es:
            vert_header = "#version 300 es\n"
            frag_header = "#version 300 es\nprecision mediump float;\n"
            logger.debug("Using OpenGL ES shader headers.")
        else:
            vert_header = "#version 330 core\n"
            frag_header = "#version 330 core\n"
            logger.debug("Using OpenGL desktop shader headers.")

        vertex_source = vert_header + vertex_source
        fragment_source = frag_header + fragment_source

        try:
            self.program = shaders.compileProgram(
                shaders.compileShader(vertex_source, GL.GL_VERTEX_SHADER),
                shaders.compileShader(fragment_source, GL.GL_FRAGMENT_SHADER),
            )
        except Exception as e:
            logger.error(f"Shader Compilation Failed: {e}", exc_info=True)
            raise

    def use(self) -> None:
        """Activates this shader program for rendering."""
        GL.glUseProgram(self.program)

    def set_mat4(self, name: str, mat: np.ndarray) -> None:
        """
        Sets a mat4 uniform in the shader.

        The matrix is expected to be in column-major format (or a transposed
        NumPy array).

        Args:
            name: The name of the uniform variable in the shader.
            mat: A 4x4 NumPy array, column-major.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mat)

    def set_mat3(self, name: str, mat: np.ndarray) -> None:
        """
        Sets a mat3 uniform in the shader.

        The matrix is expected to be in column-major format (or a transposed
        NumPy array).

        Args:
            name: The name of the uniform variable in the shader.
            mat: A 3x3 NumPy array, column-major.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE, mat)

    def set_vec2(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """
        Sets a vec2 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 2 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniform2fv(loc, 1, np.asarray(vec, dtype=np.float32))

    def set_vec3(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec3 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 3 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniform3fv(loc, 1, np.asarray(vec, dtype=np.float32))

    def set_vec4(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec4 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 4 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniform4fv(loc, 1, np.asarray(vec, dtype=np.float32))

    def cleanup(self) -> None:
        """Deletes the shader program from GPU context to free resources."""
        if getattr(self, "program", None):
            GL.glDeleteProgram(self.program)
            self.program = None

    def get_uniform_location(self, name: str) -> int:
        """Gets the location of a uniform variable."""
        return GL.glGetUniformLocation(self.program, name)

    # You could also add a set_float method for cleaner code:
    def set_float(self, name: str, value: float) -> None:
        """Sets a float uniform."""
        GL.glUniform1f(self.get_uniform_location(name), value)

    def set_int(self, name: str, value: int) -> None:
        """Sets an integer uniform."""
        GL.glUniform1i(self.get_uniform_location(name), value)


class BaseRenderer:
    """A base class for an OpenGL renderer that manages its own resources."""

    def __init__(self):
        """Initializes the resource tracking lists."""
        self.shader: Optional[Shader] = None
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

    def _cleanup_self(self) -> None:
        """
        A method for subclasses to override for their specific cleanup logic.
        """
        pass

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
        except Exception:
            logger.exception("Error during renderer cleanup")
