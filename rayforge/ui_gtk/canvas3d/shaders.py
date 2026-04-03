"""
Shader sources for the 3D workbench.
"""

SIMPLE_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aNormal;
uniform mat4 uMVP;
out vec4 vColor;
out vec3 vNormal;
out vec3 vPos;
flat out int vVertexID;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
    vNormal = aNormal;
    vPos = aPos;
    vVertexID = gl_VertexID;
}
"""

SIMPLE_FRAGMENT_SHADER = """
out vec4 FragColor;
in vec4 vColor;
in vec3 vNormal;
in vec3 vPos;
flat in int vVertexID;
uniform vec4 uColor;
uniform float uUseVertexColor;
uniform float uHasNormals;
uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform int uExecutedVertexCount;
uniform float uAlphaPending;
void main() {
    vec4 baseColor;
    if (uUseVertexColor > 0.5) {
        baseColor = vColor;
    } else {
        baseColor = uColor;
    }
    if (uHasNormals > 0.5) {
        vec3 n = normalize(vNormal);
        vec3 lightDir = normalize(uLightDir);
        float diff = max(dot(n, lightDir), 0.0);
        float ambient = 0.25;
        float diffuse = (1.0 - ambient) * diff;

        vec3 viewDir = normalize(uCameraPos - vPos);
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(n, halfDir), 0.0), 48.0);
        float specular = 0.35 * spec;

        float light = ambient + diffuse + specular;
        FragColor = vec4(baseColor.rgb * light, baseColor.a);
    } else {
        FragColor = baseColor;
    }
    if (uExecutedVertexCount >= 0) {
        if (vVertexID >= uExecutedVertexCount) {
            FragColor.a *= uAlphaPending;
        }
    }
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

# This shader calculates vertex positions relative to a single string anchor,
# ensuring the whole label billboards as one unit.
TEXT_VERTEX_SHADER = """
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
uniform float uAlpha;

void main() {
    // Sample the power value from the texture
    float power = texture(uTexture, vTexCoord).r;

    // Discard zero-power areas (make them transparent)
    if (power <= 0.0) {
        discard;
    }

    // Map power value to a color using the lookup table.
    // The second coordinate (0.5) samples the middle of the 1-pixel-high LUT
    // texture.
    vec4 color = texture(uColorLUT, vec2(power, 0.5));

    FragColor = vec4(color.rgb, color.a * uAlpha);
}
"""
