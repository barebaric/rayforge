"""
Billboarded text shader for axis/wcs labels.
"""

from .base import Shader

# This shader calculates vertex positions relative to a single string
# anchor, ensuring the whole label billboards as one unit.
TEXT_VERTEX_SHADER = """
layout (location = 0) in vec4 aVertex; // In: x, y ([-0.5, 0.5]), u, v
layout (location = 1) in vec3 aCharInfo; // In: offsetX, quadSizeX, quadHeight
layout (location = 2) in vec3 aAnchor;   // In: string anchor world position

// Uniforms
uniform mat4 uMVP;           // Model-View-Projection Matrix
uniform mat3 uBillboard;     // Camera's rotation matrix to billboard the plane

// Outputs
out vec2 vTexCoord;

void main() {
    // 1. Calculate the vertex's local position relative to the
    //    string's anchor.
    //    aVertex.x is [-0.5, 0.5], so (aVertex.x + 0.5) is [0, 1].
    //    This places the character quad correctly along the local
    //    X-axis.  The Y-position is centered on the axis.
    vec3 vertex_pos_local = vec3(
        aCharInfo.x + (aVertex.x + 0.5) * aCharInfo.y,
        aVertex.y * aCharInfo.z,
        0.0
    );

    // 2. Rotate this local position vector using the billboard matrix.
    //    This orients the entire string plane to face the camera.
    vec3 rotated_offset = uBillboard * vertex_pos_local;

    // 3. Add the final rotated offset to the string's world anchor
    //    position.
    gl_Position = uMVP * vec4(aAnchor + rotated_offset, 1.0);

    // 4. Pass texture coordinates to the fragment shader.
    vTexCoord = aVertex.zw;
}
"""

TEXT_FRAGMENT_SHADER = """
in vec2 vTexCoord;
out vec4 FragColor;

uniform sampler2D uTextAtlas;
uniform vec4 uTextColor;

void main() {
    float alpha = texture(uTextAtlas, vTexCoord).r;
    if (alpha < 0.1) {
        discard;
    }
    FragColor = vec4(uTextColor.rgb, uTextColor.a * alpha);
}
"""


class TextShader(Shader):
    """Billboarded text shader used by the axis label renderer."""

    def __init__(self):
        super().__init__(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)

    def reset_uniforms(self) -> None:
        """Sets every uniform this shader reads to its idle value."""
        self.use()
        self.set_int("uTextAtlas", 0)
        self.set_vec4("uTextColor", (1.0, 1.0, 1.0, 1.0))
