// physics/colliders.hpp
#pragma once
#include "../common/types.hpp"

enum class ColliderType : u32 {
    GroundPlane = 0,
    Box        = 1,
    Sphere     = 2, // 先不实现
    Cone       = 3  // 先不实现
};

// 简化：暂时只有一个地面 + 一个 box
struct GroundPlaneDesc {
    // 先假设法线永远是 (0,1,0)，只用高度
    f32 y = -1.5f;  // 你想调低地面的位置就在这里改
};

struct BoxDesc {
    bool enabled = false;
    Vec3 center{0.0f};      // 中心
    Vec3 half_extent{0.5f}; // 半尺寸（hx, hy, hz）
};

struct SphereDesc {
    bool enabled = false;
    Vec3 center{0.0f};
    f32  radius{0.5f};
};

struct ConeDesc {
    bool enabled = false;
    Vec3 apex{0.0f};
    Vec3 axis{0.0f, 1.0f, 0.0f};
    f32  angle{0.5f};
};

struct CollisionScene {
    GroundPlaneDesc ground;
    BoxDesc         box;

    SphereDesc sphere; // 先不在 CUDA 里实现
    ConeDesc   cone;   // 先不在 CUDA 里实现
};
