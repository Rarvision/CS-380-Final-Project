#version 450

layout(location = 0) in vec3 inNormal;
layout(location = 0) out vec4 outColor;

void main() {
    // 用位置差别来做颜色变化，增强“网格感”

    float gx = fract(gl_FragCoord.x * 0.05);
    float gy = fract(gl_FragCoord.y * 0.05);
    float grid = step(gx, 0.1) + step(gy, 0.1);

    vec3 base = vec3(0.1, 0.2, 0.6);
    vec3 line = vec3(0.9, 0.9, 1.0);

    vec3 color = mix(base, line, clamp(grid, 0.0, 1.0));
    outColor = vec4(color, 1.0);
}

