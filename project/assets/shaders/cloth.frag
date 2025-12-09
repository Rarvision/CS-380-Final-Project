#version 450

layout(location = 0) in vec3 vNormal;
layout(location = 0) out vec4 outColor;

// 和 vert 完全相同的 Push block
layout(push_constant) uniform Push {
    mat4 uMVP;
    vec4 uColor;
} pc;

void main()
{
    vec3 normal   = normalize(vNormal);
    vec3 lightDir = normalize(vec3(0.3, 1.0, 0.6));

    float diff = max(dot(normal, lightDir), 0.2);

    vec3 base = pc.uColor.rgb;   // 关键：用 push constant 传进来的颜色

    outColor = vec4(base * diff, 1.0);
}
