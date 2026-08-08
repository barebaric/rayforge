"""
Shader program classes used by the 3D workbench.
"""

from .background_shader import BackgroundShader
from .base import Shader
from .simple_shader import SimpleShader
from .text_shader import TextShader
from .texture_shader import TextureShader

__all__ = (
    "BackgroundShader",
    "Shader",
    "SimpleShader",
    "TextShader",
    "TextureShader",
)
