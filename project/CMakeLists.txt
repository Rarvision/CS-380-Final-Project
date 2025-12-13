cmake_minimum_required(VERSION 3.22)
project(ClothSim LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

# 建议：根据你的 GPU 架构设置（Ampere=86，Ada=89）
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)

# 全局 include
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/core
    ${PROJECT_SOURCE_DIR}/core/common
    ${PROJECT_SOURCE_DIR}/core/cloth
    ${PROJECT_SOURCE_DIR}/physics
    ${PROJECT_SOURCE_DIR}/physics/cuda
    ${PROJECT_SOURCE_DIR}/sim
    ${PROJECT_SOURCE_DIR}/render
    ${PROJECT_SOURCE_DIR}/render/vulkan
)

# 加入 CUDA 求解器
add_library(cuda_solver
    physics/cuda/cuda_solver.cu
    physics/cuda/cuda_solver.hpp
    physics/cuda/cuda_buffers.hpp
    physics/i_physics_solver.hpp
)

set_target_properties(cuda_solver PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# CUDA runtime
find_package(CUDAToolkit REQUIRED)
target_link_libraries(cuda_solver PUBLIC CUDA::cudart)

# 主程序
add_executable(clothsim
    app/main.cpp
    core/cloth/cloth_builder.cpp
    render/vulkan/vk_renderer.cpp
    render/vertex.hpp
    ui/ui_panel.cpp
    third_party/imgui/imgui.cpp
    third_party/imgui/imgui_draw.cpp
    third_party/imgui/imgui_widgets.cpp
    third_party/imgui/imgui_tables.cpp
    third_party/imgui/imgui_impl_glfw.cpp
    third_party/imgui/imgui_impl_vulkan.cpp
)

target_link_libraries(clothsim
    PRIVATE cuda_solver
            Vulkan::Vulkan
            glfw
)

