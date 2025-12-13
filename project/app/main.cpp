// app/main.cpp
#include <iostream>
#include <memory>
#include <vector>
#include <cfloat>

#include <GLFW/glfw3.h>

#include "../core/cloth/cloth_builder.hpp"
#include "../physics/cuda/cuda_solver.hpp"
#include "../sim/simulator.hpp"
#include "../render/vulkan/vk_renderer.hpp"
#include "../render/vertex.hpp"
#include "../ui/ui_state.hpp"
#include "../ui/ui_panel.hpp"


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

    // screen coordinate → NDC
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
        Vec3 op = p - ray_origin;
        float t = glm::dot(op, ray_dir);
        if (t < 0.0f) continue;

        Vec3 proj = ray_origin + t * ray_dir;
        Vec3 diff = p - proj;
        float d2 = glm::dot(diff, diff);

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

static void rebuild_cloth_from_ui(
    UIState& ui,
    Scene& scene,
    Simulator& sim,
    const CollisionScene& coll,
    VulkanRenderer& renderer,
    std::vector<Vec3>& host_pos,
    std::vector<Vec3>& host_normals,
    std::vector<Vertex>& host_vertices)
{
    // use UI parameters to build the cloth
    ClothBuildParams params;
    params.nx = scene.cloth.nx;
    params.ny = scene.cloth.ny;
    params.spacing  = 0.03f;
    params.mass     = ui.cloth_mass;
    params.k_struct = ui.cloth_k_struct;
    params.k_shear  = ui.cloth_k_shear;
    params.k_bend   = ui.cloth_k_bend;
    params.pin_top_edge = ui.pin_top_edge;

    scene.cloth = build_regular_grid(params);

    // initialization
    sim.init_scene(scene);
    sim.update_collision_scene(coll);

    bool releasing_pin = (!ui.pin_top_edge && ui.prev_pin_top_edge);

    ExternalForces warm_forces{};
    warm_forces.gravity  = Vec3(0.0f, -10.0f, 0.0f);
    warm_forces.wind_dir = Vec3(1.0f, 0.1f, 0.2f);

    const float WARM_DT = 1.0f / 120.0f;

    int   warm_steps;
    float warm_strength;

    if (ui.pin_top_edge) {
        warm_steps    = 120;
        warm_strength = 50.0f;
    } else {
        // warm up before actual simulation
        warm_steps    = releasing_pin ? 40 : 30;
        warm_strength = 35.0f;
    }

    warm_forces.wind_strength = warm_strength;
    sim.update_forces(warm_forces);

    for (int i = 0; i < warm_steps; ++i) {
        sim.step(WARM_DT);
    }

    warm_forces.wind_strength = 0.0f;
    sim.update_forces(warm_forces);

    // read positions/normals back from CUDA
    sim.download_positions_normals(host_pos, host_normals);

    int cloth_nx = scene.cloth.nx;
    int cloth_ny = scene.cloth.ny;

    host_vertices.resize(host_pos.size());
    for (size_t i = 0; i < host_pos.size(); ++i) {
        host_vertices[i].pos = host_pos[i];
        host_vertices[i].normal =
            (i < host_normals.size()) ? host_normals[i] : Vec3(0, 1, 0);

        int ix = static_cast<int>(i % cloth_nx);
        int iy = static_cast<int>(i / cloth_nx);

        float u = float(ix) / float(cloth_nx - 1);
        float v = float(iy) / float(cloth_ny - 1);

        host_vertices[i].uv = glm::vec2(u, v);
    }

    // update Vulkan cloth mesh
    renderer.update_mesh(host_vertices, scene.cloth.indices);

    // used to detect changing status of the slider
    ui.prev_cloth_mass     = ui.cloth_mass;
    ui.prev_cloth_k_struct = ui.cloth_k_struct;
    ui.prev_cloth_k_shear  = ui.cloth_k_shear;
    ui.prev_cloth_k_bend   = ui.cloth_k_bend;
    ui.prev_pin_top_edge   = ui.pin_top_edge;
    ui.prev_material_index = ui.current_material_index;

    // reset
    ui.request_reset_hang = false;
}

struct WindParams {
    bool  enabled       = false;
    float base_strength = 25.0f;
    float variability   = 2.0f;
    float turbulence    = 0.2f;
    Vec3  base_dir      = Vec3(1.0f, 0.05f, 0.1f);
};

struct WindRuntimeState {
    float t = 0.0f;
    float strength_noise = 1.0f;
    float dir_noise      = 0.5f;
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

    // for better wind simulation
    // combine multiple sin signal to create natural turbulence
    float n1  = std::sin(0.5f  * t);
    float n2  = std::sin(1.3f  * t + 1.7f);
    float n3  = std::sin(3.1f  * t + 0.2f);
    float raw = 0.5f * n1 + 0.3f * n2 + 0.2f * n3;

    // smoother
    const float smooth_rate = 4.0f;
    state.strength_noise += (raw - state.strength_noise) * smooth_rate * dt;

    float strength = cfg.base_strength * (1.0f + cfg.variability * state.strength_noise);

    // ±15% high frequency jitter
    float jitter = 0.15f * std::sin(6.0f * t);
    strength *= (1.0f + jitter);

    if (strength < 0.0f) strength = 0.0f;

    // slightly changed the wind direction
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

    Vec3 dir = glm::normalize(base * std::cos(angle) + right * std::sin(angle));

    forces.wind_dir      = dir;
    forces.wind_strength = strength;
}



int main()
{
    // initiliza GLFW
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

    UIState ui;
    // defult material: elastane
    ui.current_material_index = 0;
    apply_material_preset(ui);

    ui.prev_cloth_mass     = ui.cloth_mass;
    ui.prev_cloth_k_struct = ui.cloth_k_struct;
    ui.prev_cloth_k_shear  = ui.cloth_k_shear;
    ui.prev_cloth_k_bend   = ui.cloth_k_bend;
    ui.prev_pin_top_edge   = ui.pin_top_edge;
    ui.prev_material_index = ui.current_material_index;

    // VulkanRenderer scope
    {
        // create cloth scene
        ClothBuildParams params;
        params.nx = 100;
        params.ny = 50;
        params.spacing  = 0.03f;
        params.mass     = ui.cloth_mass;
        params.k_struct = ui.cloth_k_struct;
        params.k_shear  = ui.cloth_k_shear;
        params.k_bend   = ui.cloth_k_bend;
        params.pin_top_edge = ui.pin_top_edge;

        Scene scene;
        scene.type  = SceneType::HangingCloth;
        scene.cloth = build_regular_grid(params);

        // create CUDA solver + simulator
        std::unique_ptr<IPhysicsSolver> solver = std::make_unique<CudaSolver>();
        Simulator sim(std::move(solver));

        sim.init_scene(scene);

        // create collider
        CollisionScene coll;
        coll.ground.y = -0.8f;

        coll.box.enabled     = true;
        coll.box.center      = Vec3(0.0f, coll.ground.y + coll.box.half_extent.y + 0.01f, 0.0f);
        coll.box.half_extent = Vec3(0.4f, 0.4f, 0.4f);

        sim.update_collision_scene(coll);

        // initialize VulkanRenderer
        VulkanRenderer renderer;
        if (!renderer.init(window, 1600, 900)) {
            std::cerr << "Failed to init VulkanRenderer\n";
            return -1;
        }

        ui::init(window, &renderer);

        // CPU side
        std::vector<Vec3>    host_pos, host_normals;
        std::vector<Vertex>  host_vertices;

        const float FIXED_DT = 1.0f / 120.0f;
        double lastTime   = glfwGetTime();
        double accumulator = 0.0;

        WindParams wind_cfg;
        WindRuntimeState wind_state;

        // Mouse click force
        bool last_mouse_down = false;

        // uv setting
        int cloth_nx = params.nx;
        int cloth_ny = params.ny;

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

            int cloth_nx = scene.cloth.nx;
            int cloth_ny = scene.cloth.ny;

            host_vertices.resize(host_pos.size());
            for (size_t i = 0; i < host_pos.size(); ++i) {
                host_vertices[i].pos = host_pos[i];
                host_vertices[i].normal =
                    (i < host_normals.size()) ? host_normals[i] : Vec3(0, 1, 0);
                int ix = static_cast<int>(i % cloth_nx);
                int iy = static_cast<int>(i / cloth_nx);

                float u = float(ix) / float(cloth_nx - 1);
                float v = float(iy) / float(cloth_ny - 1);

                host_vertices[i].uv = glm::vec2(u, v);
            }

            renderer.update_mesh(host_vertices, scene.cloth.indices);
            renderer.update_box_mesh(coll.box.center, coll.box.half_extent * 0.94f);
            renderer.update_ground_mesh(coll.ground.y, 5.0f);
        }

        int frame_id = 0;
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            ui::new_frame();
            ui::draw_panel(ui);
            ui::end_frame(&renderer);

            // time
            double now = glfwGetTime();
            double frame_dt = now - lastTime;
            lastTime = now;
            if (frame_dt > 0.1) frame_dt = 0.1;
            accumulator += frame_dt;
            float dt = static_cast<float>(frame_dt);

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

            // mouse-click force
            bool mouse_down = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
            bool mouse_clicked = (mouse_down && !last_mouse_down);
            last_mouse_down = mouse_down;

            if (mouse_clicked && !host_pos.empty()) {
                // use ray to pick
                Ray ray = build_mouse_ray(window, renderer);

                int  hit_index = -1;
                Vec3 hit_point;
                bool has_hit = find_closest_vertex_to_ray(
                    host_pos, ray.origin, ray.dir,
                    0.25f,
                    hit_index, hit_point);

                if (has_hit) {
                    float radius   = 0.15f;
                    float strength = 30.0f;
                    sim.apply_click_impulse(hit_point, radius, strength);
                }
            }

            wind_cfg.enabled = ui.wind_enabled;
            wind_cfg.base_strength = ui.wind_base_strength;

            ExternalForces forces{};
            forces.gravity       = Vec3(0.0f, -10.0f, 0.0f);
        
            update_wind(wind_cfg, wind_state, dt, forces);

            bool cloth_params_changed =
                (ui.cloth_mass     != ui.prev_cloth_mass)     ||
                (ui.cloth_k_struct != ui.prev_cloth_k_struct) ||
                (ui.cloth_k_shear  != ui.prev_cloth_k_shear)  ||
                (ui.cloth_k_bend   != ui.prev_cloth_k_bend);

            bool pin_changed = (ui.pin_top_edge != ui.prev_pin_top_edge);

            bool need_rebuild =
                (cloth_params_changed && ui.cloth_params_dirty) ||
                pin_changed ||
                ui.request_reset_hang;

            if (need_rebuild) {
                rebuild_cloth_from_ui(
                    ui,
                    scene,
                    sim,
                    coll,
                    renderer,
                    host_pos,
                    host_normals,
                    host_vertices
                );

                ui.cloth_params_dirty = false;
            }

            // fixed physics step
            while (accumulator >= FIXED_DT) {
                sim.update_forces(forces);
                sim.step(static_cast<float>(FIXED_DT));
                accumulator -= FIXED_DT;
            }

            // read position/normals from CUDA
            sim.download_positions_normals(host_pos, host_normals);

            host_vertices.resize(host_pos.size());
            for (size_t i = 0; i < host_pos.size(); ++i) {
                host_vertices[i].pos = host_pos[i];
                host_vertices[i].normal =
                    (i < host_normals.size()) ? host_normals[i] : Vec3(0, 1, 0);
            }

            renderer.set_cloth_material(ui.current_material_index);

            Vec3 clothMin( FLT_MAX,  FLT_MAX,  FLT_MAX);
            Vec3 clothMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (const auto& p : host_pos) {
                clothMin.x = std::min(clothMin.x, p.x);
                clothMin.y = std::min(clothMin.y, p.y);
                clothMin.z = std::min(clothMin.z, p.z);
                clothMax.x = std::max(clothMax.x, p.x);
                clothMax.y = std::max(clothMax.y, p.y);
                clothMax.z = std::max(clothMax.z, p.z);
            }

            Vec3 clothCenter = (clothMin + clothMax) * 0.5f;
            Vec3 clothSize3  = (clothMax - clothMin);

            glm::vec3 lightDir = glm::normalize(glm::vec3(0.3f, 0.6f, -0.3f));

            // radius of box fake shadow
            float boxRadius = glm::length(glm::vec2(
                coll.box.half_extent.x,
                coll.box.half_extent.z
            )) * 1.3f;

            renderer.set_shadow_params(
                lightDir,
                glm::vec3(coll.box.center),
                boxRadius,
                glm::vec3(clothCenter),
                glm::vec2(clothSize3.x, clothSize3.z)
            );

            // update buffers and draw
            renderer.update_mesh(host_vertices, scene.cloth.indices);
            renderer.draw_frame();
        }

        renderer.wait_idle();
        ui::shutdown();
    }

    // destroy windows/GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
