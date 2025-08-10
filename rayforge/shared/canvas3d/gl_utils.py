"""
A collection of utility classes and functions for simplifying common
PyOpenGL tasks, such as shader compilation and buffer management.
"""

import logging
from typing import Optional, Union

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders

logger = logging.getLogger(__name__)


def gl_gen_vertex_arrays(n: int = 1) -> np.ndarray:
    """Generates one or more vertex array objects (VAOs).

    This is a wrapper around glGenVertexArrays that ensures a
    consistent return type (a NumPy array of IDs).

    Args:
        n: The number of vertex array objects to generate.

    Returns:
        A numpy array of VAO IDs.
    """
    arr_ids = GL.glGenVertexArrays(n)
    # Ensure the result is always a NumPy array for consistency.
    return np.array([arr_ids], dtype=np.uint32) if n == 1 else arr_ids


def gl_gen_buffers(n: int = 1) -> np.ndarray:
    """Generates one or more buffer objects (VBOs, EBOs, etc.).

    This is a wrapper around glGenBuffers that ensures a
    consistent return type (a NumPy array of IDs).

    Args:
        n: The number of buffer objects to generate.

    Returns:
        A numpy array of buffer IDs.
    """
    buffer_ids = GL.glGenBuffers(n)
    # Ensure the result is always a NumPy array for consistency.
    return np.array([buffer_ids], dtype=np.uint32) if n == 1 else buffer_ids


class Shader:
    """Manages a GLSL shader program, including compilation and uniforms."""

    def __init__(self, vertex_source: str, fragment_source: str):
        """Compiles and links the vertex and fragment shader sources.

        Args:
            vertex_source: The source code for the vertex shader.
            fragment_source: The source code for the fragment shader.

        Raises:
            Exception: If shader compilation or linking fails.
        """
        self._owns_shader = True
        try:
            self.program = shaders.compileProgram(
                shaders.compileShader(vertex_source, GL.GL_VERTEX_SHADER),
                shaders.compileShader(
                    fragment_source, GL.GL_FRAGMENT_SHADER
                ),
            )
        except Exception as e:
            logger.error(f"Shader Compilation Failed: {e}", exc_info=True)
            raise

    def use(self) -> None:
        """Activates this shader program for rendering."""
        GL.glUseProgram(self.program)

    def set_mat4(self, name: str, mat: np.ndarray) -> None:
        """Sets a mat4 uniform in the shader.

        The matrix is expected to be in row-major format, which is the
        default for NumPy arrays.

        Args:
            name: The name of the uniform variable in the shader.
            mat: A 4x4 NumPy array.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_TRUE, mat)

    def set_mat3(self, name: str, mat: np.ndarray) -> None:
        """Sets a mat3 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            mat: A 3x3 NumPy array.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniformMatrix3fv(loc, 1, GL.GL_TRUE, mat)

    def set_vec2(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec2 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 2 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            arr = np.asarray(vec, dtype=np.float32)
            GL.glUniform2fv(loc, 1, arr)

    def set_vec3(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec3 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 3 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            arr = np.asarray(vec, dtype=np.float32)
            GL.glUniform3fv(loc, 1, arr)

    def set_vec4(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec4 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 4 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            arr = np.asarray(vec, dtype=np.float32)
            GL.glUniform4fv(loc, 1, arr)

    def cleanup(self) -> None:
        """Deletes the shader program from GPU context to free resources."""
        if getattr(self, "program", None):
            GL.glDeleteProgram(self.program)
            self.program = None


class BaseRenderer:
    """A base class for an OpenGL renderer."""

    def __init__(self):
        """Initializes renderer state with null OpenGL object IDs."""
        self.shader: Optional[Shader] = None
        self.vao: int = 0
        self.vbo: int = 0

    def cleanup(self) -> None:
        """Cleans up all associated OpenGL resources (VAO, VBO)."""
        try:
            if self.shader:
                self.shader.cleanup()
            if self.vao:
                GL.glDeleteVertexArrays(1, (self.vao,))
            if self.vbo:
                GL.glDeleteBuffers(1, (self.vbo,))
        except Exception:
            # OpenGL context might already be destroyed.
            logger.exception("Error during renderer cleanup")
