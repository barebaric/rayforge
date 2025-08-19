SIMPLE_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

SIMPLE_FRAGMENT_SHADER = """
#version 330 core
out vec4 FragColor;
uniform vec4 uColor;
void main() {
    FragColor = uColor;
}
"""

TEXT_FRAGMENT_SHADER = """
#version 330 core
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

# This shader calculates vertex positions relative to a single string anchor,
# ensuring the whole label billboards as one unit.
TEXT_VERTEX_SHADER = """
#version 330 core
layout (location = 0) in vec4 aVertex; // In: x, y ([-0.5, 0.5]), u, v

// Uniforms
uniform mat4 uMVP;           // Model-View-Projection Matrix
uniform mat3 uBillboard;     // Camera's rotation matrix to billboard the plane
uniform vec3 uTextWorldPos;  // World position of the STRING'S anchor
uniform vec2 uQuadSize;      // Size (width, height) of the CURRENT char quad
uniform float uCharOffsetX;  // Local X-offset of the char from the anchor

// Outputs
out vec2 vTexCoord;

void main() {
    // 1. Calculate the vertex's local position relative to the string's
    //    anchor.
    //    aVertex.x is [-0.5, 0.5], so (aVertex.x + 0.5) is [0, 1].
    //    This places the character quad correctly along the local X-axis.
    //    The Y-position is centered on the axis.
    vec3 vertex_pos_local = vec3(
        uCharOffsetX + (aVertex.x + 0.5) * uQuadSize.x,
        aVertex.y * uQuadSize.y,
        0.0
    );

    // 2. Rotate this local position vector using the billboard matrix.
    //    This orients the entire string plane to face the camera.
    vec3 rotated_offset = uBillboard * vertex_pos_local;

    // 3. Add the final rotated offset to the string's world anchor position.
    gl_Position = uMVP * vec4(uTextWorldPos + rotated_offset, 1.0);

    // 4. Pass texture coordinates to the fragment shader.
    vTexCoord = aVertex.zw;
}
"""
