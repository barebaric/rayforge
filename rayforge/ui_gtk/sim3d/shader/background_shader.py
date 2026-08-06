"""
Fullscreen background gradient shader.
"""

from .base import Shader

BACKGROUND_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;

out vec2 vTexCoord;

void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    vTexCoord = aPos.xy * 0.5 + 0.5;
}
"""

BACKGROUND_FRAGMENT_SHADER = """
in vec2 vTexCoord;
out vec4 FragColor;

uniform vec3 uBgColor;
uniform vec3 uBgColorLight;

void main() {
    vec2 uv = vTexCoord;

    float vertical = mix(0.55, 1.0, uv.y);

    vec2 center = vec2(0.5, 0.45);
    float dist = length(uv - center);
    float vignette = 1.0 - smoothstep(0.0, 0.9, dist) * 0.45;

    float brightness = vertical * vignette;

    vec3 color = mix(uBgColor, uBgColorLight, brightness);

    float highlight = exp(-dist * dist * 6.0) * 0.12;
    color += vec3(highlight * 0.8, highlight * 0.9, highlight);

    FragColor = vec4(color, 1.0);
}
"""


class BackgroundShader(Shader):
    """Fullscreen gradient shader used as the canvas backdrop."""

    def __init__(self):
        super().__init__(BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER)
