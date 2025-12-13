// physics/colliders.hpp
#pragma once
#include "../common/types.hpp"

enum class ColliderType : u32 {
    GroundPlane = 0,
    Box        = 1,
    Sphere     = 2, // not implemented
    Cone       = 3  // not implemented
};

struct GroundPlaneDesc {
    f32 y = -1.5f;
};

struct BoxDesc {
    bool enabled = false;
    Vec3 center{0.0f};
    Vec3 half_extent{0.5f};
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

    SphereDesc sphere;
    ConeDesc   cone;
};
