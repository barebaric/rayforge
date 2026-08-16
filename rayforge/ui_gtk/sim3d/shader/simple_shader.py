"""
Simple two-light + LUT-driven shader used by most renderers.
"""

from .base import Shader

SIMPLE_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in vec3 aNormal;
uniform mat4 uMVP;
uniform vec3 uPartialEnd;
uniform int uPartialVertexID;
uniform float uDepthBias;
out vec4 vColor;
out vec3 vNormal;
out vec3 vPos;
flat out int vVertexID;
void main() {
    vec3 pos = aPos;
    if (gl_VertexID == uPartialVertexID) {
        pos = uPartialEnd;
    }
    gl_Position = uMVP * vec4(pos, 1.0);
    gl_Position.z += uDepthBias;
    vColor = aColor;
    vNormal = aNormal;
    vPos = pos;
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
uniform vec3 uLightDir2;
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
            float lutX = power;
            baseColor = texture(uColorLUT, vec2(lutX, lutY));
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
        float ambient = 0.35;
        float diffuse = (1.0 - ambient) * diff;

        vec3 viewDir = normalize(uCameraPos - vPos);
        vec3 halfDir = normalize(lightDir + viewDir);
        float spec = pow(max(dot(n, halfDir), 0.0), 48.0);
        float specular = 0.35 * spec;

        vec3 lightDir2 = normalize(uLightDir2);
        float diff2 = max(dot(n, lightDir2), 0.0);

        float light = ambient + diffuse + specular + 0.3 * diff2;

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


class SimpleShader(Shader):
    """The two-light LUT-driven shader used by most 3D renderers."""

    def __init__(self):
        super().__init__(SIMPLE_VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER)

    def reset_uniforms(self) -> None:
        """Sets every uniform this shader reads to its idle value."""
        self.use()
        self.set_float("uUseVertexColor", 0.0)
        self.set_float("uHasNormals", 0.0)
        self.set_int("uExecutedVertexCount", -1)
        self.set_int("uPartialVertexID", -1)
        self.set_vec3("uPartialEnd", (0.0, 0.0, 0.0))
        self.set_float("uAlphaPending", 0.2)
        self.set_float("uEmissive", 0.0)
        self.set_float("uUsePowerLUT", 0.0)
        self.set_int("uNumLaserLUTs", 1)
        self.set_vec4("uZeroPowerColor", (0.0, 0.0, 0.0, 1.0))
        self.set_float("uPointLightOn", 0.0)
        self.set_vec3("uPointLightPos", (0.0, 0.0, 0.0))
        self.set_float("uDepthBias", 0.0)
