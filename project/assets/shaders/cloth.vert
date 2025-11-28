#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 outNormal;

void main() {
    vec3 p = inPos;

    // --- 简单“假相机”：绕 X 轴旋转，把 z 抬起来 ---
    float angle = radians(60.0);    // 视角倾斜 60°
    float c = cos(angle);
    float s = sin(angle);

    // 绕 X 轴旋转：Y/Z 混合
    mat3 rotX = mat3(
        1.0, 0.0, 0.0,
        0.0,   c,  -s,
        0.0,   s,   c
    );
    p = rotX * p;

    // 缩放一下，防止超出 [-1,1]
    float scale = 0.4;     // 你可以之后微调 0.3 ~ 0.6
    p *= scale;

    gl_Position = vec4(p, 1.0);
    outNormal   = inNormal;
}
