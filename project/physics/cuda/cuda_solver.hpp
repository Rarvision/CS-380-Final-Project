#pragma once
#include <vector>

#include "../i_physics_solver.hpp"
#include "cuda_buffers.hpp"

class CudaSolver : public IPhysicsSolver {
public:
    CudaSolver();
    ~CudaSolver() override;

    void upload(const ClothModel& model) override;
    void step(f32 dt, const ExternalForces& f) override;

    void* get_device_position_buffer() override { return dev_.pos.d_ptr; }
    void* get_device_normal_buffer()   override { return dev_.normal.d_ptr; }
    void* get_device_index_buffer()    override { return dev_.indices.d_ptr; }

    void download_positions_normals(
        std::vector<Vec3>& out_pos,
        std::vector<Vec3>& out_normals) override;

private:
    CudaClothDevice dev_{};
    f32 damping_{0.01f};
    f32 ground_y_{-1.0f};

    void substep(f32 dt, const ExternalForces& f);
};
