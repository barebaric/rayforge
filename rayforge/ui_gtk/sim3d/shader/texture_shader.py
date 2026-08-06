"""
Two-source texture shader with a per-laser LUT recolour pass.
"""

from .base import Shader

TEXTURE_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 uMVP;

out vec2 vTexCoord;

void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vTexCoord = aTexCoord;
}
"""

TEXTURE_FRAGMENT_SHADER = """
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform sampler2D uColorLUT;
uniform int uNumLaserLUTs;
uniform int uLaserIndex;
uniform float uAlpha;

void main() {
    ivec2 texSize = textureSize(uTexture, 0);
    vec2 tc = vTexCoord * vec2(texSize) - 0.5;
    ivec2 base = ivec2(floor(tc));
    float power = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            ivec2 idx = clamp(
                base + ivec2(dx, dy),
                ivec2(0),
                texSize - ivec2(1)
            );
            power = max(power, texelFetch(uTexture, idx, 0).r);
        }
    }

    if (power <= 0.0) {
        discard;
    }

    float lutY = (float(uLaserIndex) + 0.5)
                 / float(max(uNumLaserLUTs, 1));
    float lutX = 0.5 + 0.5 * power;
    vec4 color = texture(uColorLUT, vec2(lutX, lutY));

    FragColor = vec4(color.rgb, color.a * uAlpha);
}
"""


class TextureShader(Shader):
    """Texture shader used by the texture-artifact renderer."""

    def __init__(self):
        super().__init__(TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER)
