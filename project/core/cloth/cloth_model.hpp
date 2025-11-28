// core/cloth/cloth_model.hpp
#pragma once
#include <vector>
#include "../common/types.hpp"

struct Spring {
    u32 i;
    u32 j;
    f32 rest;
    f32 k;
};

struct ClothModel {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
    std::vector<u32>  fixed;      // 1 = fixed, 0 = free
    std::vector<u32>  indices;    // triangles
    std::vector<Spring> springs;

    u32 nx{0}, ny{0};
    f32 mass_per_node{1.0f};

    Bounds aabb{};
};

struct ClothBuildParams {
    u32 nx{64};
    u32 ny{64};
    f32 spacing{0.02f};
    f32 mass{0.02f};
    bool pin_top_edge{true};
    f32 k_struct{8000.0f};
    f32 k_shear{4000.0f};
    f32 k_bend{0.0f};
};
