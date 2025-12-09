#pragma once
#include <memory>
#include <vector>

#include "../physics/i_physics_solver.hpp"

enum class SceneType { HangingCloth };

struct Scene {
    SceneType type{SceneType::HangingCloth};
    ClothModel cloth;
    ExternalForces forces;
};

class Simulator {
public:
    explicit Simulator(std::unique_ptr<IPhysicsSolver> solver)
        : solver_(std::move(solver)) {}

    void init_scene(const Scene& s) {
        scene_ = s;
        n_indices_ = static_cast<u32>(scene_.cloth.indices.size());
        solver_->upload(scene_.cloth);
    }

    void update_forces(const ExternalForces& f) {
        scene_.forces = f;
    }

    void step(f32 dt) {
        solver_->step(dt, scene_.forces);
    }

    // Vulkan 用的 GPU 句柄（以后做 interop 时用）
    void* device_positions() const { return solver_->get_device_position_buffer(); }
    void* device_normals() const   { return solver_->get_device_normal_buffer(); }
    void* device_indices() const   { return solver_->get_device_index_buffer(); }
    u32   index_count() const      { return n_indices_; }

    // 路线 A：CPU 回读
    void download_positions_normals(
        std::vector<Vec3>& out_pos,
        std::vector<Vec3>& out_normals)
    {
        solver_->download_positions_normals(out_pos, out_normals);
    }

    void apply_click_impulse(const Vec3& pos_ws,
                             f32 radius,
                             f32 strength)
    {
        if (solver_) {
            solver_->apply_click_impulse(pos_ws, radius, strength);
        }
    }   

private:
    std::unique_ptr<IPhysicsSolver> solver_;
    Scene scene_;
    u32   n_indices_{0};
};
