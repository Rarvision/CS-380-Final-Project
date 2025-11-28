#pragma once
#include <vector>

#include "../core/common/types.hpp"
#include "../core/cloth/cloth_model.hpp"

struct ExternalForces {
    Vec3 gravity{0.0f, -9.81f, 0.0f};
    Vec3 wind_dir{1.0f, 0.0f, 0.0f};
    f32  wind_strength{0.0f};
    bool dragging{false};
    Vec3 drag_pos_ws{0.0f};
    f32  drag_force{0.0f};
};

class IPhysicsSolver {
public:
    virtual ~IPhysicsSolver() = default;

    virtual void upload(const ClothModel& model) = 0;
    virtual void step(f32 dt, const ExternalForces& f) = 0;

    // 将来做 interop 用
    virtual void* get_device_position_buffer() = 0;
    virtual void* get_device_normal_buffer()   = 0;
    virtual void* get_device_index_buffer()    = 0;

    // 路线 A：CPU 回读给 Vulkan
    virtual void download_positions_normals(
        std::vector<Vec3>& out_pos,
        std::vector<Vec3>& out_normals) = 0;
};
