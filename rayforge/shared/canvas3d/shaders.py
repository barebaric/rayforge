"""A collection of GLSL shader sources used by the renderers."""

# A simple shader for drawing solid-colored lines and shapes.
SIMPLE_VERTEX_SHADER = """
#version 320 es
layout (location = 0) in vec3 aPos;
uniform mat4 uMVP;
void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

SIMPLE_FRAGMENT_SHADER = """
#version 320 es
precision mediump float;
out vec4 FragColor;
uniform vec4 uColor;
void main()
{
    FragColor = uColor;
}
"""

# A shader for rendering billboarded text from a texture atlas.
TEXT_VERTEX_SHADER = """
#version 320 es
// Input vertex data: vec4(position_xy, texture_coords_xy)
layout (location = 0) in vec4 aVertex;

out vec2 vTexCoords;

uniform mat4 uMVP;
uniform vec3 uTextWorldPos;
uniform vec2 uQuadSize;
uniform mat3 uBillboard;

void main()
{
    vTexCoords = aVertex.zw;

    // Rotate the quad corner positions to face the camera, then add
    // to the character's world position.
    vec3 quadPos = vec3(aVertex.xy * uQuadSize, 0.0);
    vec3 worldPos = uTextWorldPos + uBillboard * quadPos;
    gl_Position = uMVP * vec4(worldPos, 1.0);
}
"""

TEXT_FRAGMENT_SHADER = """
#version 320 es
precision mediump float;
in vec2 vTexCoords;

out vec4 FragColor;

uniform sampler2D uTexture;
uniform vec4 uTextColor;

void main()
{
    // Sample the texture atlas (red channel contains the alpha).
    float alpha = texture(uTexture, vTexCoords).r;

    // Discard transparent fragments to avoid depth-buffer issues.
    if (alpha < 0.1)
        discard;

    FragColor = vec4(uTextColor.rgb, uTextColor.a * alpha);
}
"""
