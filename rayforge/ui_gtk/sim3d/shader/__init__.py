"""
Shader program classes used by the 3D workbench.
"""

from .background_shader import BackgroundShader
from .base import Shader
from .image_shader import ImageShader
from .simple_shader import SimpleShader
from .stock_shader import StockShader
from .text_shader import TextShader
from .texture_shader import TextureShader

__all__ = (
    "BackgroundShader",
    "ImageShader",
    "Shader",
    "SimpleShader",
    "StockShader",
    "TextShader",
    "TextureShader",
)
