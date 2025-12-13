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

// 对应 C++ 的材质索引：0/1/2：布，3：方块，4：地面
const int MAT_ELASTANE    = 0;
const int MAT_HEAVY   = 1;
const int MAT_PVC = 2;
const int MAT_CUBE    = 3;
const int MAT_GROUND  = 4;

// 和碰撞里的 ground.y 一致
const float GROUND_Y = -0.8;

// set=0, binding=0，数组长度要和 C++ 里的 MAX_TEXTURES 一致（你现在是 8）
layout(set = 0, binding = 0) uniform sampler2D uTextures[8];

void main()
{
    vec3 N = normalize(vNormal);
    vec3 L = normalize(pc.uParams0.xyz);
    int  matID = int(pc.uParams0.w + 0.5);

    // ---------- 1. 基础光照 ----------
    float ndotl   = max(dot(N, L), 0.0);
    float ambient = 0.25;                      // 提高一点环境光，避免太暗
    float diff    = ambient + (1.0 - ambient) * ndotl;

    // ---------- 2. 基础颜色（带纹理） ----------
    vec4 texColor = texture(uTextures[matID], vUV);
    vec3 baseColor;
    float alpha = 1.0;

    if (matID == MAT_PVC) {
        // Pvc：不使用纹理，只给一个半透明淡蓝颜色
        baseColor = vec3(0.90, 0.96, 1.00);
        alpha     = 0.35;
    } else {
        baseColor = texColor.rgb;
    }

    // ---------- 3. 假阴影：只在地面上 ----------
    float shadow = 1.0;

    if (matID == MAT_GROUND) {
        // 限定在地面附近，并且法线基本朝上
        bool isGround =
            (abs(vWorldPos.y - GROUND_Y) < 0.05) &&
            (abs(N.y) > 0.7);

        if (isGround) {
            vec3 P = vWorldPos;

            // === 3.1 方块阴影：更明显的软圆形 ===
            vec3  boxCenter = pc.uParams1.xyz;
            float boxRadius = pc.uParams1.w;

            // 光线从 boxCenter 投到地面上的阴影中心
            float tBox = (boxCenter.y - GROUND_Y) / L.y;
            vec3  boxShadowCenter = boxCenter - L * tBox;

            vec2 dBox = P.xz - boxShadowCenter.xz;
            float rBox = length(dBox);

            // 用 inner/outer 两个半径控制阴影区域，中心更黑、边缘更柔
            float inner = boxRadius * 0.4;   // 中心基本全暗
            float outer = boxRadius * 1.1;   // 略大一点的软边

            float sBox = 1.0 - smoothstep(inner, outer, rBox);
            // 再平方一下，让中间更集中
            sBox = sBox * sBox;

            // 最暗到 0.15，比原来 0.45 深很多
            float shadowCube = mix(1.0, 0.15, sBox);

            // === 3.2 布阴影：软椭圆 ===
            vec3 clothCenter = pc.uParams2.xyz;
            vec2 clothSize   = pc.uParams3.xy;   // x,z 尺寸
            vec2 halfSize    = max(clothSize * 0.5, vec2(0.001));

            float tCloth = (clothCenter.y - GROUND_Y) / L.y;
            vec3  clothShadowCenter = clothCenter - L * tCloth;

            vec2 dC = P.xz - clothShadowCenter.xz;
            vec2 q  = dC / halfSize;            // 归一化到 [-1,1] 椭圆
            float r2 = dot(q, q);               // 椭圆内部约 0..1

            float sCloth = 1.0 - smoothstep(0.0, 1.0, r2);
            float shadowCloth = mix(1.0, 0.55, sCloth); // 布阴影稍微浅一点

            shadow = min(shadowCube, shadowCloth);
        }
    }

    // 最终光照 = 漫反射 * 阴影
    float lighting = diff * shadow;

    // 地面整体再抬一点亮度，防止太脏
    if (matID == MAT_GROUND) {
        lighting = max(lighting, 0.25);
    }

    vec3 finalColor = baseColor * lighting;
    outColor = vec4(finalColor, alpha);
}
