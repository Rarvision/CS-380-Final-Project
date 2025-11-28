// core/common/types.hpp
#pragma once
#include <cstdint>
#include <glm/glm.hpp>

using u32 = std::uint32_t;
using i32 = std::int32_t;
using f32 = float;

using Vec2 = glm::vec2;
using Vec3 = glm::vec3;

struct Bounds {
    Vec3 min{0.0f};
    Vec3 max{0.0f};
};

