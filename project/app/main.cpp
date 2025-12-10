// app/main.cpp
#include <iostream>
#include <memory>
#include <vector>

#include <GLFW/glfw3.h>

#include "../core/cloth/cloth_builder.hpp"
#include "../physics/cuda/cuda_solver.hpp"
#include "../sim/simulator.hpp"
#include "../render/vulkan/vk_renderer.hpp"
#include "../render/vertex.hpp"


struct InputState {
    bool mouse_down = false;
    double mouse_x = 0.0;
    double mouse_y = 0.0;
};

InputState input;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        InputState* in = reinterpret_cast<InputState*>(glfwGetWindowUserPointer(window));
        if (action == GLFW_PRESS) {
            in->mouse_down = true;
        } else if (action == GLFW_RELEASE) {
            in->mouse_down = false;
        }
    }
}

void cursor_pos_callback(GLFWwindow* window, double xpos, double ypos)
{
    InputState* in = reinterpret_cast<InputState*>(glfwGetWindowUserPointer(window));
    in->mouse_x = xpos;
    in->mouse_y = ypos;
}



struct Ray {
    Vec3 origin;
    Vec3 dir;
};

Ray build_mouse_ray(GLFWwindow* window,
                    const VulkanRenderer& renderer)
{
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    double mx, my;
    glfwGetCursorPos(window, &mx, &my);

    // 1) 屏幕坐标 → NDC
    glm::vec2 ndc;
    ndc.x = (2.0 * mx) / width  - 1.0;
    ndc.y = (2.0 * my) / height - 1.0;

    glm::vec4 clip_near(ndc.x, ndc.y, 0.0f, 1.0f);
    glm::vec4 clip_far (ndc.x, ndc.y, 1.0f, 1.0f);

    glm::mat4 invVP = glm::inverse(renderer.proj() * renderer.view());

    glm::vec4 world_near = invVP * clip_near;
    glm::vec4 world_far  = invVP * clip_far;
    world_near /= world_near.w;
    world_far  /= world_far.w;

    Ray ray;
    ray.origin = Vec3(world_near);
    ray.dir    = glm::normalize(Vec3(world_far - world_near));
    return ray;
}

static bool find_closest_vertex_to_ray(
    const std::vector<Vec3>& pos,
    const Vec3& ray_origin,
    const Vec3& ray_dir,
    float max_dist_ray,
    int& out_index,
    Vec3& out_hit_point)
{
    float bestDist2 = max_dist_ray * max_dist_ray;
    int   bestIdx   = -1;
    Vec3  bestPoint{0.0f};

    for (int i = 0; i < (int)pos.size(); ++i) {
        Vec3 p = pos[i];

        // 计算点 p 到射线的最近点： ray_origin + t * ray_dir
        Vec3 op = p - ray_origin;
        float t = glm::dot(op, ray_dir);
        if (t < 0.0f) continue; // 射线反向就忽略

        Vec3 proj = ray_origin + t * ray_dir; // 射线上的最近点
        Vec3 diff = p - proj;
        float d2 = glm::dot(diff, diff);      // 点到射线的距离平方

        if (d2 < bestDist2) {
            bestDist2 = d2;
            bestIdx   = i;
            bestPoint = proj;
        }
    }

    if (bestIdx >= 0) {
        out_index    = bestIdx;
        out_hit_point = bestPoint;
        return true;
    }
    return false;
}

struct WindParams {
    bool  enabled       = false;   // F 键控制
    float base_strength = 25.0f;   // 基础风力
    float variability   = 2.0f;
    float turbulence    = 0.2f;    // 方向抖动程度
    Vec3  base_dir      = Vec3(1.0f, 0.05f, 0.1f);
};

struct WindRuntimeState {
    float t = 0.0f; // 累积时间
    float strength_noise = 1.0f; // 平滑后的强度噪声
    float dir_noise      = 0.5f; // 平滑后的方向噪声
};

static void update_wind(const WindParams& cfg,
                        WindRuntimeState& state,
                        float dt,
                        ExternalForces& forces)
{
    if (!cfg.enabled) {
        forces.wind_strength = 0.0f;
        return;
    }

    state.t += dt;
    float t = state.t;

    // ---------- 1) 风的强度：更明显的“时大时小” ----------

    // 多个频率的正弦混合，做出伪随机的原始噪声 [-1,1]
    float n1  = std::sin(0.5f  * t);
    float n2  = std::sin(1.3f  * t + 1.7f);
    float n3  = std::sin(3.1f  * t + 0.2f);
    float raw = 0.5f * n1 + 0.3f * n2 + 0.2f * n3;

    // 一阶滤波，稍微平滑但变化比之前快
    const float smooth_rate = 4.0f;       // ⭐ 比 1.5 大，变化更快
    state.strength_noise += (raw - state.strength_noise) * smooth_rate * dt;

    float strength = cfg.base_strength * (1.0f + cfg.variability * state.strength_noise);

    // 再叠一点小的高频抖动（±15%）
    float jitter = 0.15f * std::sin(6.0f * t);
    strength *= (1.0f + jitter);

    if (strength < 0.0f) strength = 0.0f;

    // ---------- 2) 风向：左右轻微晃动 ----------

    float raw_dir = std::sin(0.35f * t + 2.0f);
    const float dir_smooth_rate = 2.0f;
    state.dir_noise += (raw_dir - state.dir_noise) * dir_smooth_rate * dt;

    float angle = cfg.turbulence * state.dir_noise;  // -turbulence ~ +turbulence

    Vec3 base  = glm::normalize(cfg.base_dir);
    Vec3 up    = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 right = glm::cross(up, base);

    float len2 = glm::dot(right, right);
    if (len2 < 1e-6f) {
        right = Vec3(0.0f, 0.0f, 1.0f);
    } else {
        right = glm::normalize(right);
    }

    // 在 base / right 平面内小角度转动，模拟风向左右晃
    Vec3 dir = glm::normalize(base * std::cos(angle) + right * std::sin(angle));

    forces.wind_dir      = dir;
    forces.wind_strength = strength;
}



int main()
{
    // ========== 1. 初始化 GLFW ==========
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

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(1600, 900, "ClothSim", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return -1;
    }

    InputState input;
    glfwSetWindowUserPointer(window, &input);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_pos_callback);

    // ========= 把 VulkanRenderer 放在一个局部作用域 =========
    // 关键：这样 ~VulkanRenderer 会在 glfwDestroyWindow/glfwTerminate 之前调用
    {
        // ========== 2. 构造布料场景 ==========
        ClothBuildParams params;
        params.nx = 100;
        params.ny = 50;
        params.spacing  = 0.03f;
        params.mass     = 0.02f;
        params.k_struct = 10000.0f;
        params.k_shear  = 24000.0f;
        params.k_bend   = 16000.0f;
        params.pin_top_edge = true;

        Scene scene;
        scene.type  = SceneType::HangingCloth;
        scene.cloth = build_regular_grid(params);

        // ========== 3. 创建 CUDA 求解器 + 仿真器 ==========
        std::unique_ptr<IPhysicsSolver> solver = std::make_unique<CudaSolver>();
        Simulator sim(std::move(solver));

        sim.init_scene(scene);

        // ========== 2. 构造碰撞体 ==========
        CollisionScene coll;
        coll.ground.y = -0.8f;  // 地板拉低

        coll.box.enabled     = true;
        coll.box.center      = Vec3(0.0f, coll.ground.y + coll.box.half_extent.y + 0.01f, 0.0f);
        coll.box.half_extent = Vec3(0.4f, 0.4f, 0.4f);

        sim.update_collision_scene(coll);
        

        // 将来你要球/圆锥，只需要在这里填 sphere/cone 的参数
        // coll.sphere.enabled = false;
        // coll.cone.enabled   = false;

        // // 通过 IPhysicsSolver 接口传给 CUDA
        // sim.update_collision_scene(coll);


        // ========== 4. 初始化 VulkanRenderer ==========
        VulkanRenderer renderer;
        if (!renderer.init(window, 1600, 900)) {
            std::cerr << "Failed to init VulkanRenderer\n";
            // 注意：这里直接 return 前，会先调用 ~VulkanRenderer
            return -1;
        }

        // CPU 侧缓存
        std::vector<Vec3>    host_pos, host_normals;
        std::vector<Vertex>  host_vertices;
        const auto& indices = scene.cloth.indices; // 索引在 CPU 侧固定不变

        // // 先做一小步仿真并填充一次 mesh，避免一开始 index_count=0 没东西画
        // sim.update_forces(scene.forces);
        // sim.step(1.0f / 120.0f);
        // sim.download_positions_normals(host_pos, host_normals);

        // host_vertices.resize(host_pos.size());
        // for (size_t i = 0; i < host_pos.size(); ++i) {
        //     host_vertices[i].pos = host_pos[i];
        //     host_vertices[i].normal =
        //         (i < host_normals.size()) ? host_normals[i] : Vec3(0, 1, 0);
        // }
        // renderer.update_mesh(host_vertices, indices);
        // renderer.update_box_mesh(coll.box.center, coll.box.half_extent);
        // renderer.update_ground_mesh(coll.ground.y, 5.0f);

        // ========== 5. 主循环：固定物理时间步 + 每帧更新 mesh ==========
        // ExternalForces forces{};
        // forces.gravity       = Vec3(0.0f, -10.0f, 0.0f);
        // forces.wind_dir      = Vec3(1.0f, 0.0f, 0.0f);
        // forces.wind_strength = 0.0f;   // 先不开风，之后可以用按键加风
        // forces.click_strength = 20.0f;

        const float FIXED_DT = 1.0f / 120.0f; // 物理固定步长
        double lastTime   = glfwGetTime();
        double accumulator = 0.0;

        // wind switch
        bool wind_on = false;
        bool f_down_prev = false;

        WindParams wind_cfg;
        WindRuntimeState wind_state;

        // Mouse click force
        bool last_mouse_down = false;

        // warm up before entering loop
        {
            ExternalForces warm_forces{};
            warm_forces.gravity       = Vec3(0.0f, -10.0f, 0.0f);
            warm_forces.wind_dir      = Vec3(1.0f, 0.1f, 0.2f);
            warm_forces.wind_strength = 50.0f;
            // warm_forces.click_strength = 20.0f;

            sim.update_forces(warm_forces);

            const float WARM_DT   = 1.0f / 120.0f;
            const int   WARM_STEPS = 120;
            for (int i = 0; i < WARM_STEPS; ++i) {
                sim.step(WARM_DT);
            }

            warm_forces.wind_strength = 0.0f;
            sim.update_forces(warm_forces);

            sim.download_positions_normals(host_pos, host_normals);
            host_vertices.resize(host_pos.size());
            for (size_t i = 0; i < host_pos.size(); ++i) {
                host_vertices[i].pos = host_pos[i];
                host_vertices[i].normal =
                    (i < host_normals.size()) ? host_normals[i] : Vec3(0, 1, 0);
            }

            renderer.update_mesh(host_vertices, indices);
            renderer.update_box_mesh(coll.box.center, coll.box.half_extent * 0.94f);
            renderer.update_ground_mesh(coll.ground.y, 5.0f);
        }

        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            // ---- 时间 ----
            double now = glfwGetTime();
            double frame_dt = now - lastTime;
            lastTime = now;
            if (frame_dt > 0.1) frame_dt = 0.1;  // 防止一次跳太多
            accumulator += frame_dt;
            float dt = static_cast<float>(frame_dt);

            // ---- F 键：按一下切换风开关 ----
            bool f_down = (glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS);
            if (f_down && !f_down_prev) {
                wind_on = !wind_on;
                // std::cout << "[DEBUG] wind_on = " << wind_on << "\n";
            }
            f_down_prev = f_down;

            // box control
            const float cube_speed = 1.5f;
            float move = cube_speed * dt;

            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
                coll.box.center.z += move;
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
                coll.box.center.z -= move;
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
                coll.box.center.x -= move;
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
                coll.box.center.x += move;

            coll.box.center.y = coll.ground.y + coll.box.half_extent.y;
            coll.box.center.x = glm::clamp(coll.box.center.x, -4.0f, 4.0f);
            coll.box.center.z = glm::clamp(coll.box.center.z, -4.0f, 4.0f);

            sim.update_collision_scene(coll);
            renderer.update_box_mesh(coll.box.center, coll.box.half_extent * 0.94f);

            // ---- 更新风参数（带扰动）----
            // wind_cfg.enabled = wind_on;
            // update_wind(wind_cfg, wind_state, dt, forces);

            // ---- 鼠标左键单击 → 一次性冲量 ----
            bool mouse_down = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
            bool mouse_clicked = (mouse_down && !last_mouse_down);
            last_mouse_down = mouse_down;

            if (mouse_clicked && !host_pos.empty()) {
                // 用 Ray 做 picking（用上一帧的 host_pos）
                Ray ray = build_mouse_ray(window, renderer);

                int  hit_index = -1;
                Vec3 hit_point;
                bool has_hit = find_closest_vertex_to_ray(
                    host_pos, ray.origin, ray.dir,
                    /*max_dist_ray*/ 0.25f,   // 你可以微调
                    hit_index, hit_point);

                if (has_hit) {
                    // ⭐ 直接对 CUDA 解算器施加一次“拍一下”的冲量
                    float radius   = 0.15f;   // 影响范围
                    float strength = 30.0f;   // 力度先设大一点，方便看效果
                    sim.apply_click_impulse(hit_point, radius, strength);
                }
            }

            ExternalForces forces{};
            forces.gravity       = Vec3(0.0f, -10.0f, 0.0f);
            forces.wind_dir      = Vec3(1.0f, 0.0f, 0.0f);
            forces.wind_strength = 0.0f; 
            forces.click_strength = 20.0f;

            wind_cfg.enabled = wind_on;
            update_wind(wind_cfg, wind_state, dt, forces);


            // ---- 固定步长物理子步 ----
            while (accumulator >= FIXED_DT) {
                sim.update_forces(forces);
                sim.step(static_cast<float>(FIXED_DT));
                accumulator -= FIXED_DT;

                // 如果你希望点击只作用一个子步，可以在这里关掉：
                // forces.has_click_impulse = true;
            }

            // ---- 从 CUDA 回读位置 + 法线 ----
            sim.download_positions_normals(host_pos, host_normals);

            host_vertices.resize(host_pos.size());
            for (size_t i = 0; i < host_pos.size(); ++i) {
                host_vertices[i].pos = host_pos[i];
                host_vertices[i].normal =
                    (i < host_normals.size()) ? host_normals[i] : Vec3(0, 1, 0);
            }

            // ---- 更新 Vulkan 顶点/索引缓冲并绘制 ----
            renderer.update_mesh(host_vertices, indices);
            renderer.draw_frame();
        }

        renderer.wait_idle();
        // 这里大括号结束：
        //   - renderer 在此处析构（~VulkanRenderer）
        //   - 会调用 cleanup_swapchain / cleanup_vertex_index_buffers / vkDestroyDevice / vkDestroyInstance 等
        //   - 此时 GLFW / X11 仍然是活的，所以不会出现 use-after-free
    }

    // ========== 6. 现在才销毁窗口和 GLFW ==========
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
