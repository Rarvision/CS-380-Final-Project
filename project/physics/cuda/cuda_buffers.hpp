#pragma once
#include <cstddef>
#include "../core/common/types.hpp"

struct CudaBuffer {
    void*  d_ptr{nullptr};
    size_t bytes{0};
};

struct CudaClothDevice {
    CudaBuffer pos;
    CudaBuffer vel;
    CudaBuffer normal;
    CudaBuffer indices;
    CudaBuffer springs;
    CudaBuffer fixed;

    u32 n_vertices{0};
    u32 n_indices{0};
    u32 n_springs{0};
};
