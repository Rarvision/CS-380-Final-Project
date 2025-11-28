// physics/cuda/cuda_solver.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <vector>
#include <stdexcept>
#include <cstring>

#include "cuda_solver.hpp"
#include "cuda_buffers.hpp"
#include "../core/cloth/cloth_model.hpp"
#include "../core/common/types.hpp"

// 为 device 使用简单 float3 运算
__device__ inline float3 f3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ inline float3 f3_sub(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ inline float3 f3_scale(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ inline float f3_dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ inline float f3_len(const float3& a) {
    return sqrtf(f3_dot(a,a));
}
__device__ inline float3 f3_cross(const float3& a, const float3& b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

struct SpringDev {
    u32 i;
    u32 j;
    f32 rest;
    f32 k;
};

// 全局力缓冲
static CudaBuffer d_force;

// ---------------- host 辅助函数 ----------------

template<typename T>
static void alloc_cuda_buffer(CudaBuffer& buf, size_t count) {
    buf.bytes = count * sizeof(T);
    cudaError_t err = cudaMalloc(&buf.d_ptr, buf.bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed");
    }
}

template<typename T>
static void upload_vector(CudaBuffer& buf, const std::vector<T>& v) {
    if (v.empty()) {
        buf.d_ptr = nullptr;
        buf.bytes = 0;
        return;
    }
    alloc_cuda_buffer<T>(buf, v.size());
    cudaMemcpy(buf.d_ptr, v.data(), buf.bytes, cudaMemcpyHostToDevice);
}

// ---------------- kernels ----------------

__global__ void kernel_clear_forces(float3* force, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    force[idx] = make_float3(0,0,0);
}

__device__ inline void atomicAddFloat3(float3* addr, const float3& v)
{
    atomicAdd(&(addr->x), v.x);
    atomicAdd(&(addr->y), v.y);
    atomicAdd(&(addr->z), v.z);
}

__global__ void kernel_apply_springs(
    float3* pos, SpringDev* springs, u32 n_springs, float3* force)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_springs) return;

    SpringDev s = springs[idx];
    float3 pi = pos[s.i];
    float3 pj = pos[s.j];
    float3 dir = f3_sub(pj, pi);
    float len = f3_len(dir);
    if (len < 1e-6f) return;

    float3 n = f3_scale(dir, 1.0f / len);
    float x = len - s.rest;
    float3 f = f3_scale(n, s.k * x);

    atomicAddFloat3(&force[s.i],  f);
    atomicAddFloat3(&force[s.j],  f3_scale(f, -1.0f));
}

__global__ void kernel_apply_wind(float3* force, u32 n, float3 wind_dir, float strength)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    force[idx] = f3_add(force[idx], f3_scale(wind_dir, strength));
}

__global__ void kernel_apply_drag(
    float3* pos, float3* force, u32 n,
    float3 drag_pos, float strength, float radius)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float3 p = pos[idx];
    float3 d = f3_sub(drag_pos, p);
    float dist = f3_len(d);
    if (dist < radius && dist > 1e-6f) {
        float3 dir = f3_scale(d, 1.0f / dist);
        float w = (radius - dist) / radius;
        force[idx] = f3_add(force[idx], f3_scale(dir, strength * w));
    }
}

__global__ void kernel_integrate(
    float3* pos, float3* vel, float3* force,
    const u32* fixed, u32 n_vertices,
    float inv_mass, float dt, float3 gravity, float damping)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vertices) return;

    if (fixed[idx]) {
        vel[idx] = make_float3(0,0,0);
        return;
    }

    float3 a = f3_add(gravity, f3_scale(force[idx], inv_mass));
    float3 v = vel[idx];
    v = f3_scale(f3_add(v, f3_scale(a, dt)), (1.0f - damping));
    float3 p = f3_add(pos[idx], f3_scale(v, dt));

    vel[idx] = v;
    pos[idx] = p;
}

__global__ void kernel_collide_plane(float3* pos, float3* vel, u32 n, float ground_y)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float3 p = pos[idx];
    if (p.y < ground_y) {
        p.y = ground_y;
        vel[idx].y = 0.0f;
        pos[idx] = p;
    }
}

__global__ void kernel_clear_normals(float3* normal, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    normal[idx] = make_float3(0,0,0);
}

__global__ void kernel_accumulate_normals(
    float3* pos, float3* normal, const u32* indices, u32 n_indices)
{
    u32 triIdx = blockIdx.x * blockDim.x + threadIdx.x;
    u32 idx = triIdx * 3;
    if (idx + 2 >= n_indices) return;

    u32 i0 = indices[idx + 0];
    u32 i1 = indices[idx + 1];
    u32 i2 = indices[idx + 2];

    float3 p0 = pos[i0];
    float3 p1 = pos[i1];
    float3 p2 = pos[i2];

    float3 e1 = f3_sub(p1, p0);
    float3 e2 = f3_sub(p2, p0);
    float3 n  = f3_cross(e1, e2);

    atomicAddFloat3(&normal[i0], n);
    atomicAddFloat3(&normal[i1], n);
    atomicAddFloat3(&normal[i2], n);
}

__global__ void kernel_normalize_normals(float3* normal, u32 n)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float3 v = normal[idx];
    float len = f3_len(v);
    if (len > 1e-6f) {
        normal[idx] = f3_scale(v, 1.0f / len);
    } else {
        normal[idx] = make_float3(0,1,0);
    }
}

// ---------------- CudaSolver 方法实现 ----------------

CudaSolver::CudaSolver()
{
    d_force.d_ptr = nullptr;
    d_force.bytes = 0;
}

CudaSolver::~CudaSolver()
{
    auto free_buf = [](CudaBuffer& b) {
        if (b.d_ptr) cudaFree(b.d_ptr);
        b.d_ptr = nullptr;
        b.bytes = 0;
    };
    free_buf(dev_.pos);
    free_buf(dev_.vel);
    free_buf(dev_.normal);
    free_buf(dev_.indices);
    free_buf(dev_.springs);
    free_buf(dev_.fixed);
    free_buf(d_force);
}

void CudaSolver::upload(const ClothModel& model)
{
    dev_.n_vertices = static_cast<u32>(model.positions.size());
    dev_.n_springs  = static_cast<u32>(model.springs.size());
    dev_.n_indices  = static_cast<u32>(model.indices.size());

    // pos / vel -> float3
    std::vector<float3> h_pos(dev_.n_vertices);
    std::vector<float3> h_vel(dev_.n_vertices);
    for (u32 i = 0; i < dev_.n_vertices; ++i) {
        const Vec3& p = model.positions[i];
        const Vec3& v = model.velocities[i];
        h_pos[i] = make_float3(p.x, p.y, p.z);
        h_vel[i] = make_float3(v.x, v.y, v.z);
    }
    upload_vector<float3>(dev_.pos, h_pos);
    upload_vector<float3>(dev_.vel, h_vel);
    alloc_cuda_buffer<float3>(dev_.normal, dev_.n_vertices);

    // indices / fixed
    upload_vector<u32>(dev_.indices, model.indices);
    upload_vector<u32>(dev_.fixed, model.fixed);

    // springs
    std::vector<SpringDev> springs_dev(dev_.n_springs);
    for (u32 i = 0; i < dev_.n_springs; ++i) {
        const Spring& s = model.springs[i];
        springs_dev[i] = { s.i, s.j, s.rest, s.k };
    }
    upload_vector<SpringDev>(dev_.springs, springs_dev);

    // force buffer
    alloc_cuda_buffer<float3>(d_force, dev_.n_vertices);
}

void CudaSolver::substep(f32 dt, const ExternalForces& f)
{
    const u32 nVerts   = dev_.n_vertices;
    const u32 nSprings = dev_.n_springs;
    const u32 nIndices = dev_.n_indices;

    float3* d_pos   = reinterpret_cast<float3*>(dev_.pos.d_ptr);
    float3* d_vel   = reinterpret_cast<float3*>(dev_.vel.d_ptr);
    float3* d_norm  = reinterpret_cast<float3*>(dev_.normal.d_ptr);
    u32*    d_idx   = reinterpret_cast<u32*>(dev_.indices.d_ptr);
    u32*    d_fixed = reinterpret_cast<u32*>(dev_.fixed.d_ptr);
    float3* d_f     = reinterpret_cast<float3*>(d_force.d_ptr);

    const int blockSize = 256;
    int gridVerts   = (nVerts   + blockSize - 1) / blockSize;
    int gridSprings = (nSprings + blockSize - 1) / blockSize;
    int gridTris    = (nIndices / 3 + blockSize - 1) / blockSize;

    // 1. 清力
    kernel_clear_forces<<<gridVerts, blockSize>>>(d_f, nVerts);

    // 2. 风 / 拖拽
    if (f.wind_strength > 0.0f) {
        float3 wdir = make_float3(f.wind_dir.x, f.wind_dir.y, f.wind_dir.z);
        kernel_apply_wind<<<gridVerts, blockSize>>>(d_f, nVerts, wdir, f.wind_strength);
    }
    if (f.dragging && f.drag_force > 0.0f) {
        float3 dragPos = make_float3(f.drag_pos_ws.x, f.drag_pos_ws.y, f.drag_pos_ws.z);
        kernel_apply_drag<<<gridVerts, blockSize>>>(
            d_pos, d_f, nVerts, dragPos, f.drag_force, 0.2f);
    }

    // 3. 弹簧
    kernel_apply_springs<<<gridSprings, blockSize>>>(
        d_pos, reinterpret_cast<SpringDev*>(dev_.springs.d_ptr), nSprings, d_f);

    // 4. 积分
    const f32 inv_mass = 1.0f; // 以后可以用 model.mass_per_node
    float3 g = make_float3(f.gravity.x, f.gravity.y, f.gravity.z);
    kernel_integrate<<<gridVerts, blockSize>>>(
        d_pos, d_vel, d_f, d_fixed, nVerts, inv_mass, dt, g, damping_);

    // 5. 碰撞
    kernel_collide_plane<<<gridVerts, blockSize>>>(d_pos, d_vel, nVerts, ground_y_);

    // 6. 法线
    kernel_clear_normals<<<gridVerts, blockSize>>>(d_norm, nVerts);
    kernel_accumulate_normals<<<gridTris, blockSize>>>(
        d_pos, d_norm, d_idx, nIndices);
    kernel_normalize_normals<<<gridVerts, blockSize>>>(d_norm, nVerts);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ")
                                 + cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA sync failed: ")
                                 + cudaGetErrorString(err));
    }
}

void CudaSolver::step(f32 dt, const ExternalForces& f)
{
    const int substeps = 2;
    f32 h = dt / substeps;
    for (int i = 0; i < substeps; ++i) {
        substep(h, f);
    }
}

void CudaSolver::download_positions_normals(
    std::vector<Vec3>& out_pos,
    std::vector<Vec3>& out_normals)
{
    out_pos.resize(dev_.n_vertices);
    out_normals.resize(dev_.n_vertices);

    std::vector<float3> tmp_pos(dev_.n_vertices);
    std::vector<float3> tmp_norm(dev_.n_vertices);

    float3* d_pos  = reinterpret_cast<float3*>(dev_.pos.d_ptr);
    float3* d_norm = reinterpret_cast<float3*>(dev_.normal.d_ptr);

    cudaMemcpy(tmp_pos.data(),  d_pos,  dev_.n_vertices * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp_norm.data(), d_norm, dev_.n_vertices * sizeof(float3), cudaMemcpyDeviceToHost);

    for (u32 i = 0; i < dev_.n_vertices; ++i) {
        out_pos[i]    = Vec3(tmp_pos[i].x,  tmp_pos[i].y,  tmp_pos[i].z);
        out_normals[i]= Vec3(tmp_norm[i].x,tmp_norm[i].y,tmp_norm[i].z);
    }
}
