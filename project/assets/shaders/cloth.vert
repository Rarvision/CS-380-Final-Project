#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 vNormal;

layout(push_constant) uniform Push {
    mat4 uMVP;
} pc;

void main()
{
    gl_Position = pc.uMVP * vec4(inPos, 1.0);
    // 简单起见：假设没有非均匀缩放，直接把模型空间法线传出去
    vNormal = inNormal;
}
