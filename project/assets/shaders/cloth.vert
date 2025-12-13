#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec2 vUV;
layout(location = 2) out vec3 vWorldPos;

layout(push_constant) uniform Push {
    mat4 uMVP;
    vec4 uParams0;
    vec4 uParams1;
    vec4 uParams2;
    vec4 uParams3;
} pc;

void main()
{
    vNormal   = inNormal;
    vUV       = inUV;
    vWorldPos = inPos;

    gl_Position = pc.uMVP * vec4(inPos, 1.0);
}
