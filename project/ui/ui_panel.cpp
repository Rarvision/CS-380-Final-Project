// ui/ui_panel.cpp
#include "ui_panel.hpp"
#include "ui_state.hpp"

#include "../third_party/imgui/imgui.h"
#include "../third_party/imgui/imgui_impl_glfw.h"
#include "../third_party/imgui/imgui_impl_vulkan.h"

#include "../render/vulkan/vk_renderer.hpp"

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include <iostream>

namespace ui
{

static VkDescriptorPool g_imgui_descriptor_pool = VK_NULL_HANDLE;
static VkDevice         g_imgui_device          = VK_NULL_HANDLE;

void init(GLFWwindow* window, VulkanRenderer* renderer)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    // backend: GLFW
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // seperate descriptor pool for ImGUI
    VkDevice device = renderer->vk_device();
    g_imgui_device  = device;

    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER,                1000 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,          1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,   1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,   1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,         1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,         1000 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT,       1000 },
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.maxSets       = 1000u * (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.poolSizeCount = (uint32_t)IM_ARRAYSIZE(pool_sizes);
    pool_info.pPoolSizes    = pool_sizes;

    if (vkCreateDescriptorPool(device, &pool_info, nullptr, &g_imgui_descriptor_pool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ImGui descriptor pool");
    }

    // ImGui_ImplVulkan_InitInfo
    ImGui_ImplVulkan_InitInfo init_info{};
    init_info.ApiVersion        = VK_API_VERSION_1_1;
    init_info.Instance          = renderer->vk_instance();
    init_info.PhysicalDevice    = renderer->vk_physical_device();
    init_info.Device            = renderer->vk_device();
    init_info.QueueFamily       = renderer->graphics_queue_family_index();
    init_info.Queue             = renderer->vk_graphics_queue();
    init_info.DescriptorPool    = g_imgui_descriptor_pool;
    init_info.DescriptorPoolSize = 0;

    uint32_t image_count = renderer->swapchain_image_count();
    init_info.MinImageCount = image_count;
    init_info.ImageCount    = image_count;

    init_info.PipelineCache = VK_NULL_HANDLE;

    // tell backend the render pass for main viewport
    init_info.PipelineInfoMain.RenderPass  = renderer->vk_render_pass();
    init_info.PipelineInfoMain.Subpass     = 0;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    init_info.UseDynamicRendering = false;

    init_info.Allocator        = nullptr;
    init_info.CheckVkResultFn  = nullptr;
    init_info.MinAllocationSize = 0;

    init_info.CustomShaderVertCreateInfo = {};
    init_info.CustomShaderFragCreateInfo = {};

    ImGui_ImplVulkan_Init(&init_info);
}

void shutdown()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    if (g_imgui_descriptor_pool != VK_NULL_HANDLE && g_imgui_device != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(g_imgui_device, g_imgui_descriptor_pool, nullptr);
        g_imgui_descriptor_pool = VK_NULL_HANDLE;
    }

    ImGui::DestroyContext();
}

void new_frame()
{
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
}

static void slider_with_release(const char* label,
                                float* v,
                                float v_min,
                                float v_max,
                                const char* fmt,
                                UIState& ui,
                                ImGuiSliderFlags flags = 0)
{
    ImGui::SliderFloat(label, v, v_min, v_max, fmt, flags);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        ui.cloth_params_dirty = true;
    }
}

void draw_panel(UIState& ui_state)
{
    ImGui::Begin("Cloth Controls");

    // 1) 风场
    if (ImGui::CollapsingHeader("Wind", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Enabled", &ui_state.wind_enabled);
        ImGui::SliderFloat("Base Strength", &ui_state.wind_base_strength,
                           0.0f, 100.0f, "%.1f");
    }

    // 2) 布料物理参数
    if (ImGui::CollapsingHeader("Cloth Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
        slider_with_release("Mass",
                            &ui_state.cloth_mass,
                            0.005f, 0.1f, "%.3f",
                            ui_state);

        slider_with_release("k_struct",
                            &ui_state.cloth_k_struct,
                            1000.0f, 50000.0f, "%.0f",
                            ui_state,
                            ImGuiSliderFlags_Logarithmic);

        slider_with_release("k_shear",
                            &ui_state.cloth_k_shear,
                            1000.0f, 50000.0f, "%.0f",
                            ui_state,
                            ImGuiSliderFlags_Logarithmic);

        slider_with_release("k_bend",
                            &ui_state.cloth_k_bend,
                            1000.0f, 50000.0f, "%.0f",
                            ui_state,
                            ImGuiSliderFlags_Logarithmic);
    }

    // 3) 悬挂 / 掉落
    if (ImGui::CollapsingHeader("Pinning", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Checkbox("Pin top edge", &ui_state.pin_top_edge);

        ImGui::TextWrapped("When pin is ON, cloth is hanging.\n"
                           "Turn it OFF to let the cloth fall.\n"
                           "Click the button below to reset to hanging pose.");

        if (ImGui::Button("Reset cloth to hanging pose")) {
            ui_state.request_reset_hang = true;
        }
    }

    // 4) 材质预设
    if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
        int index = ui_state.current_material_index;
        if (ImGui::Combo("Cloth Material", &index,
                         ui_state.material_names, IM_ARRAYSIZE(ui_state.material_names))) {
            ui_state.current_material_index = index;
            apply_material_preset(ui_state);
            ui_state.cloth_params_dirty = true;
        }

        ImGui::Text("Current preset:");
        ImGui::Text("mass     = %.4f", ui_state.cloth_mass);
        ImGui::Text("k_struct = %.0f", ui_state.cloth_k_struct);
        ImGui::Text("k_shear  = %.0f", ui_state.cloth_k_shear);
        ImGui::Text("k_bend   = %.0f", ui_state.cloth_k_bend);
    }

    ImGui::End();
}

void end_frame(VulkanRenderer* /*renderer*/)
{
    ImGui::Render();
    ImDrawData* draw_data = ImGui::GetDrawData();
}

} // namespace ui
