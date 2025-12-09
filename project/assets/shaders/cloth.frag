#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 0) out vec4 outColor;

void main()
{
    // ========= 1. 法线处理 =========
    vec3 N = normalize(vNormal);

    // 我们把「你看到的那一面比较亮」当成“正面”
    // 从你刚才的观察来看，这一面其实是 !gl_FrontFacing
    bool isFront = !gl_FrontFacing;

    // 为了让正反两面在光照上都平滑，
    // 让法线始终朝向“视觉上的正面”这一侧
    if (!isFront) {
        N = -N;
    }

    vec3 L = normalize(vec3(0.3, 0.7, 0.4));
    float ndotl = dot(N, L);

    // Half-Lambert：让背光也有一点亮度，不会完全掉到灰
    float diff = ndotl * 0.5 + 0.5; // [-1,1] -> [0,1]
    diff = clamp(diff, 0.0, 1.0);
    diff = diff * diff;            // 稍微柔一点

    float ambient = 0.2;
    float lighting = ambient + 0.8 * diff;

    vec3 baseColor = vec3(0.85, 0.85, 0.9);
    vec3 color = baseColor * lighting;
    outColor = vec4(color, 1.0);
}
