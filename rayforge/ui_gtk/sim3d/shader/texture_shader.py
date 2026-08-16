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
uniform float uMaxMip;

void main() {
    // Select the mip level that matches the texel footprint so the
    // raster rows do not alias into a moire banding pattern when the
    // texture is minified (zoomed out).
    ivec2 texSize = textureSize(uTexture, 0);
    vec2 p = vTexCoord * vec2(texSize);
    vec2 ddx = abs(dFdx(p));
    vec2 ddy = abs(dFdy(p));
    float footprint = max(max(ddx.x, ddx.y), max(ddy.x, ddy.y));
    float lod = clamp(log2(max(footprint, 1.0)), 0.0, uMaxMip);
    int li = int(lod + 0.5);
    ivec2 ls = textureSize(uTexture, li);

    vec2 tc = vTexCoord * vec2(ls) - 0.5;
    ivec2 base = ivec2(floor(tc));
    float power = 0.0;
    for (int dy = 0; dy <= 1; dy++) {
        for (int dx = 0; dx <= 1; dx++) {
            ivec2 idx = clamp(
                base + ivec2(dx, dy),
                ivec2(0),
                ls - ivec2(1)
            );
            power = max(power, texelFetch(uTexture, idx, li).r);
        }
    }

    if (power <= 0.0) {
        // Write depth for the gaps between scanlines so geometry drawn
        // behind the raster (e.g. rotary module models) cannot bleed
        // through the texture's zero-power pixels and band the preview.
        // Contribute no colour (alpha 0).
        FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    float lutY = (float(uLaserIndex) + 0.5)
                 / float(max(uNumLaserLUTs, 1));
    float lutX = power;
    vec4 color = texture(uColorLUT, vec2(lutX, lutY));

    FragColor = vec4(color.rgb, color.a * uAlpha);
}
"""


class TextureShader(Shader):
    """Texture shader used by the texture-artifact renderer."""

    def __init__(self):
        super().__init__(TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER)

    def reset_uniforms(self) -> None:
        """Sets every uniform this shader reads to its idle value."""
        self.use()
        self.set_int("uTexture", 0)
        self.set_int("uColorLUT", 1)
        self.set_int("uNumLaserLUTs", 1)
        self.set_int("uLaserIndex", 0)
        self.set_float("uAlpha", 1.0)
        self.set_float("uMaxMip", 0.0)
