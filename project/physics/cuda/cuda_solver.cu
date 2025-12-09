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
__host__ __device__ inline float3 make_f3(float x, float y, float z) {
    return make_float3(x, y, z);
}

__host__ __device__ inline float3 make_f3(const Vec3& v) {
    return make_float3(v.x, v.y, v.z);
}

__host__ __device__ inline Vec3 to_vec3(const float3& f) {
    return Vec3(f.x, f.y, f.z);
}

// ======= float3 基础运算 =======
__device__ inline float3 f3_add(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 f3_sub(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 f3_scale(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ inline float  f3_dot(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ inline float3 f3_cross(const float3& a, const float3& b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__device__ inline float  f3_len(const float3& a) {
    return sqrtf(f3_dot(a,a));
}

__device__ inline float3 f3_normalize(const float3& a) {
    float len = f3_len(a);
    if (len < 1e-6f) return make_float3(0.f, 0.f, 0.f);
    float inv = 1.0f / len;
    return f3_scale(a, inv);
}

// ======= float3 的 atomicAdd =======
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

__global__ void kernel_apply_wind(
    float3* pos,
    float3* normal,
    float3* force,
    u32 n,
    float3 wind_dir,
    float strength,
    float time)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // 归一化风向
    float3 wdir = f3_normalize(wind_dir);

    float3 p   = pos[idx];
    float3 nrm = f3_normalize(normal[idx]);

    // ---------- 1) 只对“迎风面”施力 ----------
    float ndot = f3_dot(nrm, wdir);              // 法线与风向的夹角
    float facing = fmaxf(ndot, 0.0f);            // 背风面不受风/受很小风

    // 基础风力（方向 = 风向）
    float3 F_base = f3_scale(wdir, strength * facing);

    // ---------- 2) 空间 + 时间噪声，制造旗子上的“涟漪” ----------
    // 频率可以调节：freq 控制波长；time_freq 控制时间变化速度
    const float freq      = 8.0f;    // 空间频率：越大图案越密
    const float time_freq = 1.5f;    // 时间频率：越大变化越快

    float phase = freq * (p.x + 0.5f * p.y) + time_freq * time;
    float noise = __sinf(phase) * __cosf(0.7f * phase + 1.3f); // [-1,1] 左右

    // 让噪声幅值小一些，避免把布吹炸
    const float noise_scale = 0.3f;  // 占整体强度的大约 30%
    float  noise_strength   = strength * noise_scale * noise;

    // 噪声方向：取一个垂直于风向的向量，让布有“横向摆动”
    float3 up    = make_float3(0.0f, 1.0f, 0.0f);
    float3 side  = f3_cross(wdir, up);
    if (f3_len(side) < 1e-4f) {
        // 避免 wdir 跟 up 平行
        side = f3_cross(wdir, make_float3(0.0f, 0.0f, 1.0f));
    }
    side = f3_normalize(side);

    float3 F_turb = f3_scale(side, noise_strength);

    // ---------- 3) 合成风力 ----------
    float3 F_total = f3_add(F_base, F_turb);
    force[idx]     = f3_add(force[idx], F_total);
}


__global__ void kernel_click_impulse(
    float3* pos, float3* vel, u32 n,
    float3 click_pos, float strength, float radius)
{
    u32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float3 p = pos[idx];
    float3 d = f3_sub(p, click_pos); // 从点击中心指向粒子（往外推）
    float dist = f3_len(d);
    if (dist < radius && dist > 1e-6f) {
        // 基于距离的权重：中心最大，边缘为 0
        float w = (radius - dist) / radius; // 0..1

        // 单位方向
        float3 dir = f3_scale(d, 1.0f / dist);

        // 原本的冲量大小
        float impulse = strength * w;

        // ⭐ 限制单点最大速度增量，防止一次性拉太猛
        const float max_dv = 4.0f;
        if (impulse > max_dv) impulse = max_dv;

        float3 dv = f3_scale(dir, impulse);
        vel[idx] = f3_add(vel[idx], dv);
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
    time_ = 0.0f;
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

    // 分配 pos / vel / normal / force 都按 float3 计
    alloc_cuda_buffer<float3>(dev_.pos, dev_.n_vertices);
    alloc_cuda_buffer<float3>(dev_.vel, dev_.n_vertices);
    alloc_cuda_buffer<float3>(dev_.normal, dev_.n_vertices);
    alloc_cuda_buffer<float3>(d_force, dev_.n_vertices);

    // host -> device: Vec3 数组转 float3 数组
    std::vector<float3> tmp_pos(dev_.n_vertices);
    std::vector<float3> tmp_vel(dev_.n_vertices, make_float3(0.f, 0.f, 0.f));
    for (u32 i = 0; i < dev_.n_vertices; ++i) {
        tmp_pos[i] = make_f3(model.positions[i]);
        // 如果你 ClothModel 里暂时没用 velocities，可以先全 0
        // 否则 tmp_vel[i] = make_f3(model.velocities[i]);
    }
    cudaMemcpy(dev_.pos.d_ptr, tmp_pos.data(),
               dev_.n_vertices * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_.vel.d_ptr, tmp_vel.data(),
               dev_.n_vertices * sizeof(float3), cudaMemcpyHostToDevice);

    // indices / fixed 可以直接按 u32 传
    upload_vector<u32>(dev_.indices, model.indices);
    upload_vector<u32>(dev_.fixed,   model.fixed);

    // springs 同前，只是保持 SpringDev 类型
    std::vector<SpringDev> springs_dev(dev_.n_springs);
    for (u32 i = 0; i < dev_.n_springs; ++i) {
        const Spring& s = model.springs[i];
        springs_dev[i] = { s.i, s.j, s.rest, s.k };
    }
    upload_vector<SpringDev>(dev_.springs, springs_dev);
}

void CudaSolver::apply_click_impulse(const Vec3& pos_ws,
                                     f32 radius,
                                     f32 strength)
{
    if (dev_.n_vertices == 0) return;

    float3* d_pos = reinterpret_cast<float3*>(dev_.pos.d_ptr);
    float3* d_vel = reinterpret_cast<float3*>(dev_.vel.d_ptr);

    const int blockSize = 256;
    int gridVerts = (dev_.n_vertices + blockSize - 1) / blockSize;

    float3 cpos = make_f3(pos_ws);
    kernel_click_impulse<<<gridVerts, blockSize>>>(
        d_pos, d_vel, dev_.n_vertices, cpos, strength, radius);
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
    // SpringDev* d_s  = reinterpret_cast<SpringDev*>(dev_.springs.d_ptr);
    float3* d_f     = reinterpret_cast<float3*>(d_force.d_ptr);

    const int blockSize = 256;
    int gridVerts   = (nVerts   + blockSize - 1) / blockSize;
    int gridSprings = (nSprings + blockSize - 1) / blockSize;
    int gridTris    = (nIndices / 3 + blockSize - 1) / blockSize;

    time_ += dt;
    // 1. 清力
    kernel_clear_forces<<<gridVerts, blockSize>>>(d_f, nVerts);

    // 2. 风
    if (f.wind_strength > 0.0f) {
    float3 wdir = make_f3(f.wind_dir);  // 你已经有 make_f3(Vec3) 了
    kernel_apply_wind<<<gridVerts, blockSize>>>(
        d_pos, d_norm, d_f,
        nVerts,
        wdir,
        f.wind_strength,
        time_);
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

    cudaMemcpy(tmp_pos.data(), dev_.pos.d_ptr,
               dev_.n_vertices * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp_norm.data(), dev_.normal.d_ptr,
               dev_.n_vertices * sizeof(float3), cudaMemcpyDeviceToHost);

    for (u32 i = 0; i < dev_.n_vertices; ++i) {
        out_pos[i]    = to_vec3(tmp_pos[i]);
        out_normals[i]= to_vec3(tmp_norm[i]);
    }
}
