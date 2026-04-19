"""
Shader sources for the 3D workbench.
"""

SIMPLE_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in float aVertexIndex;
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
    vVertexID = int(aVertexIndex);
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
uniform float uEmissive;
uniform vec3 uPointLightPos;
uniform float uPointLightOn;
uniform float uUsePowerLUT;
uniform sampler2D uColorLUT;
uniform int uNumLaserLUTs;
uniform vec4 uZeroPowerColor;
void main() {
    vec4 baseColor;
    if (uUsePowerLUT > 0.5) {
        float power = clamp(vColor.r, 0.0, 1.0);
        if (power < 0.001) {
            baseColor = uZeroPowerColor;
        } else {
            int laserIdx = int(vColor.g + 0.5);
            float lutY = (float(laserIdx) + 0.5)
                         / float(max(uNumLaserLUTs, 1));
            baseColor = texture(uColorLUT, vec2(power, lutY));
        }
    } else if (uUseVertexColor > 0.5) {
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

        if (uPointLightOn > 0.5) {
            // Point light from laser
            vec3 toPoint = uPointLightPos - vPos;
            float dist = length(toPoint);
            float atten = 1.0 / (1.0 + 0.005 * dist * dist);
            if (dist > 0.001) {
                vec3 plDir = toPoint / dist;
                float plDiff = max(dot(n, plDir), 0.0);
                light += plDiff * atten;
            }
        }

        FragColor = vec4(baseColor.rgb * light, baseColor.a);
    } else {
        FragColor = baseColor;
    }
    FragColor.rgb *= (1.0 + uEmissive);
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
    vec4 color = texture(uColorLUT, vec2(power, lutY));

    FragColor = vec4(color.rgb, color.a * uAlpha);
}
"""
