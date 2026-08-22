"""
PBR shader for solid stock meshes (Cook-Torrance + analytic IBL).

Direct lighting uses two directional lights plus the laser point
light.  Indirect lighting uses the split-sum approximation: a
precomputed 32x32 BRDF integration LUT (computed in Rust by
:func:`raygeo.image.pbr.generate_brdf_lut` and uploaded by
:class:`~rayforge.ui_gtk.sim3d.renderer.stock_renderer.StockRenderer`
as an RG16F texture) combined with an analytic "studio environment"
whose irradiance is a hemispheric sky/ground gradient, so no cubemap
is needed.

The stock albedo comes from a GL_SRGB8_ALPHA8 texture (sampled values
are linearised by the hardware) or, for materials without a texture,
from a solid sRGB colour converted to linear in the shader.  Output is
Reinhard-tonemapped and gamma-encoded to match the display-ready
values the other canvas shaders emit.
"""

import numpy as np

from .base import Shader

STOCK_VERTEX_SHADER = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aUV;
layout (location = 3) in vec2 aPowerUV;
uniform mat4 uMVP;
uniform mat4 uModel;
out vec3 vNormal;
out vec3 vLocalNormal;
out vec3 vWorldPos;
out vec2 vUV;
out vec2 vPowerUV;
void main() {
    vWorldPos = (uModel * vec4(aPos, 1.0)).xyz;
    // The stock transform is a pure translation (world -> visual), so
    // the upper-left 3x3 safely maps normals without a transpose of
    // the inverse.
    vNormal = normalize(mat3(uModel) * aNormal);
    vLocalNormal = aNormal;
    vUV = aUV;
    vPowerUV = aPowerUV;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

STOCK_FRAGMENT_SHADER = """
out vec4 FragColor;
in vec3 vNormal;
in vec3 vLocalNormal;
in vec3 vWorldPos;
in vec2 vUV;
in vec2 vPowerUV;
uniform vec3 uCameraPos;
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uLightDir2;
uniform vec3 uLightColor2;
uniform vec3 uPointLightPos;
uniform float uPointLightOn;
uniform vec3 uPointLightColor;
uniform vec4 uAlbedo;
uniform float uRoughness;
uniform float uMetallic;
uniform float uUseTexture;
uniform float uAlpha;
uniform sampler2D uTexture;
uniform sampler2D uBrdfLut;
uniform sampler2D uPowerTexture;
uniform float uUsePowerTexture;
uniform float uRotary;
uniform vec3 uAmbientSky;
uniform vec3 uAmbientGround;
uniform vec3 uTint;
uniform float uUseTint;

const float PI = 3.14159265359;

vec3 srgb_to_linear(vec3 c) {
    return pow(c, vec3(2.2));
}

vec3 linear_to_srgb(vec3 c) {
    return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2));
}

float distribution_ggx(vec3 n, vec3 h, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float ndh = max(dot(n, h), 0.0);
    float ndh2 = ndh * ndh;
    float denom = (ndh2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;
    return a2 / max(denom, 1e-7);
}

float geometry_schlick_ggx(float ndv, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    return ndv / (ndv * (1.0 - k) + k);
}

float geometry_smith(vec3 n, vec3 v, vec3 l, float roughness) {
    float ndv = max(dot(n, v), 0.0);
    float ndl = max(dot(n, l), 0.0);
    float ggx2 = geometry_schlick_ggx(ndv, roughness);
    float ggx1 = geometry_schlick_ggx(ndl, roughness);
    return ggx1 * ggx2;
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(
        clamp(1.0 - cos_theta, 0.0, 1.0), 5.0
    );
}

vec3 direct_light(
    vec3 n, vec3 v, vec3 l, vec3 radiance,
    vec3 albedo, float roughness, float metallic, vec3 f0
) {
    vec3 h = normalize(v + l);
    float ndf = distribution_ggx(n, h, roughness);
    float g = geometry_smith(n, v, l, roughness);
    vec3 f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    vec3 numerator = ndf * g * f;
    float denominator = 4.0 * max(dot(n, v), 0.0)
                        * max(dot(n, l), 0.0);
    vec3 specular = numerator / max(denominator, 1e-7);
    vec3 kd = (vec3(1.0) - f) * (1.0 - metallic);
    return (kd * albedo / PI + specular)
           * radiance * max(dot(n, l), 0.0);
}

// Irradiance of the fitted studio environment: a hemispheric
// sky/ground gradient along +z (the engrave plane normal).
vec3 env_irradiance(vec3 n) {
    return mix(uAmbientGround, uAmbientSky, 0.5 + 0.5 * n.z);
}

// Burn transfer: the laser PWM fraction shaped by a noise floor and a
// sub-unity exponent, mapping typical engraving power to
// mostly-charred. Near-zero power (image blacks, laser off) stays at 0.
float burn_transfer(vec2 uv) {
    float power = texture(uPowerTexture, uv).r;
    float shaped = clamp((power - 0.05) / 0.95, 0.0, 1.0);
    return pow(shaped, 0.45);
}

void main() {
    vec3 n = normalize(vNormal);
    vec3 v = normalize(uCameraPos - vWorldPos);

    vec4 albedo_tex = texture(uTexture, vUV);
    // Colorize: shift the texture to the tint hue, preserving its
    // per-pixel brightness (luma * tint). Applied in linear space.
    if (uUseTint > 0.5) {
        float luma = dot(
            albedo_tex.rgb, vec3(0.2126, 0.7152, 0.0722)
        );
        albedo_tex.rgb = luma * uTint;
    }
    vec3 albedo = mix(
        srgb_to_linear(uAlbedo.rgb), albedo_tex.rgb, uUseTexture
    );

    float roughness = clamp(uRoughness, 0.045, 1.0);

    // Laser burn-in: the folded surface map chars the material. Power
    // maps carry the laser PWM fraction (often 0.2-0.6 for engraving);
    // a noise floor plus a sub-unity exponent maps that range to
    // mostly-charred, and the char also absorbs light, so the lit
    // result is attenuated beyond the albedo mix. Near-zero power
    // (image blacks, laser off) does not char.
    //
    // The burn is confined to the engraved faces, tested on the local
    // normal so cylinder kinematics don't affect the test. Flat stock
    // is engraved on its top face (local normal +z); rotary stock on
    // its lateral surface (normals radial to the local x axis), not
    // the end caps.
    float burn = 0.0;
    vec3 char_albedo = albedo;
    bool burn_face = uRotary > 0.5
        ? abs(vLocalNormal.x) < 0.5
        : vLocalNormal.z > 0.5;
    if (uUsePowerTexture > 0.5 && burn_face) {
        vec2 ts = 1.0 / vec2(textureSize(uPowerTexture, 0));

        burn = burn_transfer(vPowerUV);

        // Char colour ramps from warm scorch at low power, through
        // near-black char, to cool ash when over-burned.
        vec3 scorch = vec3(0.14, 0.07, 0.02);
        vec3 char_dark = vec3(0.02, 0.012, 0.006);
        vec3 ash = vec3(0.11, 0.11, 0.13);
        vec3 char_color;
        if (burn < 0.5) {
            char_color = mix(scorch, char_dark, burn * 2.0);
        } else {
            char_color = mix(char_dark, ash, (burn - 0.5) * 2.0);
        }
        char_albedo = mix(albedo, char_color, burn);
        roughness = clamp(mix(roughness, 0.55, burn), 0.045, 1.0);

        // Soft heat-affected halo: a wider warm-brown fringe around
        // the sharp char, strongest at the burn boundary.
        float halo = 0.0;
        for (int x = -1; x <= 1; x++) {
            for (int y = -1; y <= 1; y++) {
                halo += texture(
                    uPowerTexture, vPowerUV + vec2(float(x), float(y)) * ts
                ).r;
            }
        }
        halo = clamp((halo / 9.0 - 0.05) / 0.95, 0.0, 1.0);
        halo = pow(halo, 0.8) * (1.0 - burn);
        char_albedo = mix(char_albedo, scorch * 1.3, 0.45 * halo);
    }
    albedo = char_albedo;

    vec3 f0 = mix(vec3(0.04), albedo, uMetallic);

    vec3 color = vec3(0.0);

    // Key and fill directional lights.
    vec3 l1 = normalize(uLightDir);
    color += direct_light(
        n, v, l1, uLightColor, albedo, roughness, uMetallic, f0
    );
    vec3 l2 = normalize(uLightDir2);
    color += direct_light(
        n, v, l2, uLightColor2, albedo, roughness, uMetallic, f0
    );

    // Laser point light with distance attenuation.
    if (uPointLightOn > 0.5) {
        vec3 to_point = uPointLightPos - vWorldPos;
        float dist = length(to_point);
        float atten = 1.0 / (1.0 + 0.005 * dist * dist);
        if (dist > 0.001) {
            color += direct_light(
                n, v, to_point / dist, uPointLightColor * atten,
                albedo, roughness, uMetallic, f0
            );
        }
    }

    // Split-sum IBL: diffuse irradiance plus specular via the
    // precomputed BRDF LUT and the environment's radiance estimate
    // along the reflection vector.
    vec3 f_ambient = fresnel_schlick(max(dot(n, v), 0.0), f0);
    vec3 kd = (vec3(1.0) - f_ambient) * (1.0 - uMetallic);
    vec3 diffuse_ibl = kd * albedo * env_irradiance(n);

    vec3 r = reflect(-v, n);
    vec3 env_radiance = mix(uAmbientGround, uAmbientSky, 0.5 + 0.5 * r.z);
    env_radiance *= 1.0
        + 0.5 * pow(clamp(r.z, 0.0, 1.0), 3.0);
    vec2 brdf = texture(
        uBrdfLut, vec2(max(dot(n, v), 0.0), roughness)
    ).rg;
    vec3 specular_ibl = (f0 * brdf.x + brdf.y) * env_radiance;

    color += diffuse_ibl + specular_ibl;

    // Charred material absorbs light overall, not just diffuse
    // albedo — attenuate the lit result by the burn factor. Kept mild
    // so the bump-driven self-shadowing (from the perturbed normal)
    // does most of the "carved groove" shading.
    color *= 1.0 - 0.25 * burn;

    // Tonemap and encode for the display-ready framebuffer.  ACES
    // Filmic (Narkowicz 2015) keeps darks dark and rolls off
    // highlights, unlike Reinhard which lifts midtones and washed the
    // dark stock out toward grey.
    vec3 mapped = clamp(
        (color * (2.51 * color + 0.03))
            / (color * (2.43 * color + 0.59) + 0.14),
        0.0, 1.0
    );
    FragColor = vec4(linear_to_srgb(mapped), uAlbedo.a * uAlpha);
}
"""


class StockShader(Shader):
    """Cook-Torrance PBR shader with analytic split-sum IBL."""

    def __init__(self):
        super().__init__(STOCK_VERTEX_SHADER, STOCK_FRAGMENT_SHADER)

    def reset_uniforms(self) -> None:
        """Sets every uniform this shader reads to its idle value."""
        self.use()
        self.set_vec3("uCameraPos", (0.0, 0.0, 0.0))
        self.set_vec3("uLightDir", (0.5, 0.8, 1.0))
        self.set_vec3("uLightColor", (1.0, 0.97, 0.93))
        self.set_vec3("uLightDir2", (-0.6, -0.4, 0.3))
        self.set_vec3("uLightColor2", (0.25, 0.28, 0.33))
        self.set_vec3("uPointLightPos", (0.0, 0.0, 0.0))
        self.set_float("uPointLightOn", 0.0)
        self.set_vec3("uPointLightColor", (2.0, 1.1, 0.6))
        self.set_vec4("uAlbedo", (1.0, 1.0, 1.0, 1.0))
        self.set_float("uRoughness", 0.8)
        self.set_float("uMetallic", 0.0)
        self.set_float("uUseTexture", 0.0)
        self.set_float("uAlpha", 1.0)
        self.set_int("uTexture", 0)
        self.set_int("uBrdfLut", 1)
        self.set_int("uPowerTexture", 2)
        self.set_float("uUsePowerTexture", 0.0)
        self.set_float("uRotary", 0.0)
        self.set_vec3("uTint", (1.0, 1.0, 1.0))
        self.set_float("uUseTint", 0.0)
        self.set_vec3("uAmbientSky", (0.18, 0.20, 0.24))
        self.set_vec3("uAmbientGround", (0.10, 0.09, 0.08))
        self.set_mat4("uModel", np.eye(4, dtype=np.float32))
        self.set_mat4("uMVP", np.eye(4, dtype=np.float32))
