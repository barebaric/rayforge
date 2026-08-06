"""
Base class for GLSL shader programs.

Compilation, uniform setting, and the snapshot/restore pair used by
the ``gl_state`` / ``uniform_block`` context managers.
"""

import logging
from typing import Any, Union

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
        # Cache of the most recent value written to each uniform by
        # ``set_*``.  ``save()`` / ``restore()`` use this to snapshot
        # and replay uniforms across a renderer pass without re-issuing
        # GL queries.
        self._uniform_values: dict[str, Any] = {}
        # Determine the correct GLSL header for the current context.
        version_str = GL.glGetString(GL.GL_VERSION)
        is_es = version_str is not None and b"OpenGL ES" in version_str
        if is_es:
            vert_header = "#version 300 es\n"
            frag_header = (
                "#version 300 es\n"
                "precision highp float;\n"
                "precision highp int;\n"
            )
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
        except shaders.ShaderValidationError as e:
            logger.warning(
                "Shader validation failed during program creation; "
                "retrying without validation: %s",
                e,
            )
            self.program = shaders.compileProgram(
                shaders.compileShader(vertex_source, GL.GL_VERTEX_SHADER),
                shaders.compileShader(fragment_source, GL.GL_FRAGMENT_SHADER),
                validate=False,
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

        The matrix is expected to be in column-major format (or a
        transposed NumPy array).

        Args:
            name: The name of the uniform variable in the shader.
            mat: A 4x4 NumPy array, column-major.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniformMatrix4fv(loc, 1, GL.GL_FALSE, mat)
            self._uniform_values[name] = ("mat4", np.array(mat, copy=True))

    def set_mat3(self, name: str, mat: np.ndarray) -> None:
        """
        Sets a mat3 uniform in the shader.

        The matrix is expected to be in column-major format (or a
        transposed NumPy array).

        Args:
            name: The name of the uniform variable in the shader.
            mat: A 3x3 NumPy array, column-major.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniformMatrix3fv(loc, 1, GL.GL_FALSE, mat)
            self._uniform_values[name] = ("mat3", np.array(mat, copy=True))

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
            self._uniform_values[name] = (
                "vec2",
                np.asarray(vec, dtype=np.float32).copy(),
            )

    def set_vec3(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec3 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 3 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniform3fv(loc, 1, np.asarray(vec, dtype=np.float32))
            self._uniform_values[name] = (
                "vec3",
                np.asarray(vec, dtype=np.float32).copy(),
            )

    def set_vec4(self, name: str, vec: Union[tuple, list, np.ndarray]) -> None:
        """Sets a vec4 uniform in the shader.

        Args:
            name: The name of the uniform variable in the shader.
            vec: A sequence (tuple, list, or array) of 4 floats.
        """
        loc = GL.glGetUniformLocation(self.program, name)
        if loc != -1:
            GL.glUniform4fv(loc, 1, np.asarray(vec, dtype=np.float32))
            self._uniform_values[name] = (
                "vec4",
                np.asarray(vec, dtype=np.float32).copy(),
            )

    def save(self) -> dict[str, Any]:
        """
        Snapshots all ``set_*``-tracked uniform values.

        Returns the snapshot so the caller can replay it later via
        :meth:`restore`.  Used by the ``uniform_block`` companion of
        ``gl_state`` to bracket renderers that mutate overlapping
        uniforms.
        """
        return {
            name: (kind, np.array(val, copy=True))
            for name, (kind, val) in self._uniform_values.items()
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        """
        Replays a uniform snapshot produced by :meth:`save`.

        Re-issues the same ``set_*`` call for each entry, so values are
        restored even if a renderer clobbered them mid-frame.  Unknown
        uniform locations are silently skipped (as ``set_*`` is).
        """
        for name, (kind, val) in snapshot.items():
            if kind == "mat4":
                self.set_mat4(name, val)
            elif kind == "mat3":
                self.set_mat3(name, val)
            elif kind == "vec2":
                self.set_vec2(name, val)
            elif kind == "vec3":
                self.set_vec3(name, val)
            elif kind == "vec4":
                self.set_vec4(name, val)
            elif kind == "float":
                self.set_float(name, val)
            elif kind == "int":
                self.set_int(name, val)

    def cleanup(self) -> None:
        """Deletes the shader program from GPU context to free resources."""
        if getattr(self, "program", None):
            GL.glDeleteProgram(self.program)
            self.program = None

    def get_uniform_location(self, name: str) -> int:
        """Gets the location of a uniform variable."""
        return GL.glGetUniformLocation(self.program, name)

    def set_float(self, name: str, value: float) -> None:
        """Sets a float uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            GL.glUniform1f(loc, value)
            self._uniform_values[name] = ("float", float(value))

    def set_int(self, name: str, value: int) -> None:
        """Sets an integer uniform."""
        loc = self.get_uniform_location(name)
        if loc != -1:
            GL.glUniform1i(loc, value)
            self._uniform_values[name] = ("int", int(value))
