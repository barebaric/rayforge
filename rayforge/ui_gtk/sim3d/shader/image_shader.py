"""
Unlit RGBA texture shader for workpiece base images.

Samples the image texture directly (no laser colour LUT) so the 3D
canvas shows the workpiece's source image the same way the 2D canvas
does.
"""

from .base import Shader

IMAGE_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 uMVP;

out vec2 vTexCoord;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vTexCoord = aTexCoord;
}
"""

IMAGE_FRAGMENT_SHADER = """
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform float uAlpha;

void main() {
    vec4 color = texture(uTexture, vTexCoord);
    FragColor = vec4(color.rgb, color.a * uAlpha);
}
"""


class ImageShader(Shader):
    """Unlit shader that draws a plain RGBA texture quad."""

    def __init__(self):
        super().__init__(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)

    def reset_uniforms(self) -> None:
        """Sets every uniform this shader reads to its idle value."""
        self.use()
        self.set_int("uTexture", 0)
        self.set_float("uAlpha", 1.0)
