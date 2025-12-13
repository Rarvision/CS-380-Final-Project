#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec2 vUV;
layout(location = 2) in vec3 vWorldPos;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    mat4 uMVP;
    vec4 uParams0;   // xyz = lightDir, w = materialIndex
    vec4 uParams1;   // xyz = boxCenter, w = boxRadius
    vec4 uParams2;   // xyz = clothCenter, w = 1.0
    vec4 uParams3;   // xy  = clothSize.xz
} pc;

const int MAT_ELASTANE    = 0;
const int MAT_COTTON   = 1;
const int MAT_PVC = 2;
const int MAT_CUBE    = 3;
const int MAT_GROUND  = 4;

const float GROUND_Y = -0.8;

layout(set = 0, binding = 0) uniform sampler2D uTextures[8];

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(pc.uParams0.xyz);
    int  matID = int(pc.uParams0.w + 0.5);

    // basic lighting
    float ndotl   = max(dot(N, L), 0.0);
    float ambient = 0.25;
    float diff    = ambient + (1.0 - ambient) * ndotl;

    // basic color
    vec4 texColor = texture(uTextures[matID], vUV);
    vec3 baseColor;
    float alpha = 1.0;

    if (matID == MAT_PVC) {
        // PVC doesn't use texture
        baseColor = vec3(0.90, 0.96, 1.00);
        alpha     = 0.35;
    } else {
        baseColor = texColor.rgb;
    }

    // fake shadow
    float shadow = 1.0;

    if (matID == MAT_GROUND) {
        bool isGround =
            (abs(vWorldPos.y - GROUND_Y) < 0.05) &&
            (abs(N.y) > 0.7);

        if (isGround) {
            vec3 P = vWorldPos;

            // box shadow
            vec3  boxCenter = pc.uParams1.xyz;
            float boxRadius = pc.uParams1.w;

            float tBox = (boxCenter.y - GROUND_Y) / L.y;
            vec3  boxShadowCenter = boxCenter - L * tBox;

            vec2 dBox = P.xz - boxShadowCenter.xz;
            float rBox = length(dBox);

            // inner -> blacker outer -> softer
            float inner = boxRadius * 0.4;
            float outer = boxRadius * 1.1;

            float sBox = 1.0 - smoothstep(inner, outer, rBox);
            sBox = sBox * sBox;

            float shadowCube = mix(1.0, 0.15, sBox);

            // cloth shadow
            vec3 clothCenter = pc.uParams2.xyz;
            vec2 clothSize   = pc.uParams3.xy;
            vec2 halfSize    = max(clothSize * 0.5, vec2(0.001));

            float tCloth = (clothCenter.y - GROUND_Y) / L.y;
            vec3  clothShadowCenter = clothCenter - L * tCloth;

            vec2 dC = P.xz - clothShadowCenter.xz;
            vec2 q  = dC / halfSize;
            float r2 = dot(q, q);

            float sCloth = 1.0 - smoothstep(0.0, 1.0, r2);
            float shadowCloth = mix(1.0, 0.55, sCloth);

            shadow = min(shadowCube, shadowCloth);
        }
    }

    float lighting = diff * shadow;

    if (matID == MAT_GROUND) {
        lighting = max(lighting, 0.25);
    }

    vec3 finalColor = baseColor * lighting;
    outColor = vec4(finalColor, alpha);
}
