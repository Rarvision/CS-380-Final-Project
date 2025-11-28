// app/main.cpp
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

#include <GLFW/glfw3.h>

#include "../core/cloth/cloth_builder.hpp"
#include "../physics/cuda/cuda_solver.hpp"
#include "../sim/simulator.hpp"
#include "../render/vulkan/vk_renderer.hpp"
#include "../render/vertex.hpp"

int main()
{
    // 初始化 GLFW
    auto glfw_error_callback = [](int error, const char* description) {
    std::cerr << "[GLFW ERROR] (" << error << "): "
              << (description ? description : "unknown") << "\n";
    };

    glfwSetErrorCallback(glfw_error_callback);

    if (!glfwInit()) {
        const char* desc = nullptr;
        int code = glfwGetError(&desc);
        std::cerr << "Failed to init GLFW, code = " << code
                  << ", msg = " << (desc ? desc : "unknown") << "\n";
        return -1;
    }


    // 告诉 GLFW 我们用 Vulkan，不创建 OpenGL context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1600, 900, "ClothSim", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    // 1. 构造布料场景
    ClothBuildParams params;
    params.nx = 40;
    params.ny = 40;
    params.spacing = 0.03f;
    params.mass = 0.02f;
    params.k_struct = 8000.0f;
    params.k_shear  = 4000.0f;
    params.k_bend   = 2000.0f;
    params.pin_top_edge = true;

    Scene scene;
    scene.type  = SceneType::HangingCloth;
    scene.cloth = build_regular_grid(params);

    // 2. 创建 CUDA 求解器 + 仿真器
    std::unique_ptr<IPhysicsSolver> solver = std::make_unique<CudaSolver>();
    Simulator sim(std::move(solver));
    sim.init_scene(scene);

    // 3. 初始化 VulkanRenderer
    VulkanRenderer renderer;
    if (!renderer.init(window, 1600, 900)) {
        std::cerr << "Failed to init VulkanRenderer\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    // CPU 侧缓存
    std::vector<Vec3> host_pos, host_normals;
    std::vector<Vertex> host_vertices;
    const auto& indices = scene.cloth.indices; // CPU 上索引不变

    // 初始填充一次避免空 buffer
    sim.update_forces(scene.forces);
    sim.step(1.0f / 120.0f);
    sim.download_positions_normals(host_pos, host_normals);
    host_vertices.resize(host_pos.size());
    for (size_t i = 0; i < host_pos.size(); ++i) {
        host_vertices[i].pos    = host_pos[i];
        host_vertices[i].normal = (i < host_normals.size() ? host_normals[i] : Vec3(0,1,0));
    }
    renderer.update_mesh(host_vertices, indices);

    // 主循环
    ExternalForces forces;
    forces.gravity = Vec3(0.0f, -9.81f, 0.0f);
    forces.wind_dir = Vec3(1.0f, 0.0f, 0.0f);
    forces.wind_strength = 0.0f;

    const f32 dt = 1.0f / 120.0f;

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // TODO: 这里可以根据键盘/鼠标改 forces（例如按 key 添加风）

        sim.update_forces(forces);
        sim.step(dt);

        // CPU 回读
        sim.download_positions_normals(host_pos, host_normals);
        host_vertices.resize(host_pos.size());
        for (size_t i = 0; i < host_pos.size(); ++i) {
            host_vertices[i].pos    = host_pos[i];
            host_vertices[i].normal = (i < host_normals.size() ? host_normals[i] : Vec3(0,1,0));
        }
        renderer.update_mesh(host_vertices, indices);
        renderer.draw_frame();
    }

    renderer.wait_idle();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
