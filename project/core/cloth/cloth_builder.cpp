// core/cloth/cloth_builder.cpp
#include "cloth_builder.hpp"
#include <algorithm>
#include <cmath>

ClothModel build_regular_grid(const ClothBuildParams& p)
{
    ClothModel c;
    c.nx = p.nx;
    c.ny = p.ny;
    const u32 n_vertices = p.nx * p.ny;
    c.positions.resize(n_vertices);
    c.velocities.assign(n_vertices, Vec3(0.0f));
    c.fixed.assign(n_vertices, 0u);
    c.mass_per_node = p.mass;

    // 简单生成一个位于 XZ 平面、Y 为 0 的网格，上边缘在 +Z
    const f32 w = (p.nx - 1) * p.spacing;
    const f32 h = (p.ny - 1) * p.spacing;
    const Vec3 origin = Vec3(-0.5f * w, 0.0f, 0.0f);

    for (u32 y = 0; y < p.ny; ++y) {
        for (u32 x = 0; x < p.nx; ++x) {
            const u32 idx = y * p.nx + x;
            f32 px = origin.x + x * p.spacing;
            f32 py = origin.y;                // 初始平面
            f32 pz = origin.z + y * p.spacing;
            c.positions[idx] = Vec3(px, py, pz);
        }
    }

    // 三角索引（两三角组成一个 quad）
    for (u32 y = 0; y < p.ny - 1; ++y) {
        for (u32 x = 0; x < p.nx - 1; ++x) {
            u32 i0 = y * p.nx + x;
            u32 i1 = y * p.nx + (x + 1);
            u32 i2 = (y + 1) * p.nx + x;
            u32 i3 = (y + 1) * p.nx + (x + 1);

            // 约定同向（三角剖分）
            c.indices.push_back(i0);
            c.indices.push_back(i2);
            c.indices.push_back(i1);

            c.indices.push_back(i1);
            c.indices.push_back(i2);
            c.indices.push_back(i3);
        }
    }

    if (p.pin_top_edge) {
        tag_pins_top_edge(c);
    }
    compute_springs(c, p.k_struct, p.k_shear, p.k_bend);

    // 简单 AABB
    c.aabb.min = c.aabb.max = c.positions[0];
    for (auto& v : c.positions) {
        c.aabb.min = glm::min(c.aabb.min, v);
        c.aabb.max = glm::max(c.aabb.max, v);
    }

    return c;
}

void tag_pins_top_edge(ClothModel& c)
{
    // “顶边”：y = ny - 1 行
    for (u32 x = 0; x < c.nx; ++x) {
        u32 idx = (c.ny - 1) * c.nx + x;
        c.fixed[idx] = 1u;
    }
}

static void add_spring(ClothModel& c, u32 i, u32 j, f32 k)
{
    Vec3 pi = c.positions[i];
    Vec3 pj = c.positions[j];
    f32 rest = glm::length(pj - pi);
    Spring s{ i, j, rest, k };
    c.springs.push_back(s);
}

void compute_springs(ClothModel& c, f32 k_struct, f32 k_shear, f32 k_bend)
{
    c.springs.clear();

    // 结构弹簧（上下左右）
    for (u32 y = 0; y < c.ny; ++y) {
        for (u32 x = 0; x < c.nx; ++x) {
            u32 i = y * c.nx + x;
            if (x + 1 < c.nx) {
                u32 j = y * c.nx + (x + 1);
                add_spring(c, i, j, k_struct);
            }
            if (y + 1 < c.ny) {
                u32 j = (y + 1) * c.nx + x;
                add_spring(c, i, j, k_struct);
            }
        }
    }

    // 剪切弹簧（对角线）
    for (u32 y = 0; y < c.ny - 1; ++y) {
        for (u32 x = 0; x < c.nx - 1; ++x) {
            u32 i = y * c.nx + x;
            u32 i_right = y * c.nx + (x + 1);
            u32 i_down = (y + 1) * c.nx + x;
            u32 i_down_right = (y + 1) * c.nx + (x + 1);

            add_spring(c, i, i_down_right, k_shear);
            add_spring(c, i_right, i_down, k_shear);
        }
    }

    // 弯曲弹簧（间隔一个点的结构弹簧）
    if (k_bend > 0.0f) {
        for (u32 y = 0; y < c.ny; ++y) {
            for (u32 x = 0; x < c.nx; ++x) {
                u32 i = y * c.nx + x;
                if (x + 2 < c.nx) {
                    u32 j = y * c.nx + (x + 2);
                    add_spring(c, i, j, k_bend);
                }
                if (y + 2 < c.ny) {
                    u32 j = (y + 2) * c.nx + x;
                    add_spring(c, i, j, k_bend);
                }
            }
        }
    }
}
