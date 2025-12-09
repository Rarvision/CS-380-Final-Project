#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 vNormal;

// 注意：顶点 & 片元里 Push 的定义必须完全一致
layout(push_constant) uniform Push {
    mat4 uMVP;
    vec4 uColor;
} pc;

void main()
{
    vNormal = inNormal;
    gl_Position = pc.uMVP * vec4(inPos, 1.0);

    gl_Position.z -= 1e-4 * gl_Position.w;
}
