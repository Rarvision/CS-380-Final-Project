#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include <cstdint>

#include "../core/common/types.hpp"
#include "../render/vertex.hpp"

struct GLFWwindow;

class VulkanRenderer {
public:
    VulkanRenderer() = default;
    ~VulkanRenderer();

    const glm::mat4& view() const { return view_; }
    const glm::mat4& proj() const { return proj_; }
    // glm::mat4 mvp_ = glm::mat4(1.0f);

    bool init(GLFWwindow* window, uint32_t width, uint32_t height);
    void update_mesh(const std::vector<Vertex>& vertices,
                     const std::vector<u32>& indices);
    void draw_frame();
    void wait_idle();

private:
    GLFWwindow* window_{nullptr};

    VkInstance       instance_{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device_{VK_NULL_HANDLE};
    VkDevice         device_{VK_NULL_HANDLE};
    VkQueue          graphics_queue_{VK_NULL_HANDLE};
    VkQueue          present_queue_{VK_NULL_HANDLE};
    VkSurfaceKHR     surface_{VK_NULL_HANDLE};

    VkSwapchainKHR              swapchain_{VK_NULL_HANDLE};
    std::vector<VkImage>        swapchain_images_;
    std::vector<VkImageView>    swapchain_image_views_;
    VkFormat                    swapchain_image_format_{};
    VkExtent2D                  swapchain_extent_{};

    VkRenderPass      render_pass_{VK_NULL_HANDLE};
    VkPipelineLayout  pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline        graphics_pipeline_{VK_NULL_HANDLE};

    std::vector<VkFramebuffer>  framebuffers_;

    VkCommandPool     command_pool_{VK_NULL_HANDLE};
    std::vector<VkCommandBuffer> command_buffers_;

    // 同步对象
    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence>     in_flight_fences_;
    size_t current_frame_{0};
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    // 顶点/索引缓冲（host-visible 简版）
    VkBuffer       vertex_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory vertex_buffer_memory_{VK_NULL_HANDLE};
    size_t         vertex_buffer_size_{0};

    VkBuffer       index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory index_buffer_memory_{VK_NULL_HANDLE};
    size_t         index_count_{0};

    size_t index_buffer_size_{0};

    //
    bool           cb_recorded_{false};

    // MVP
    glm::mat4 proj_{1.0f};
    glm::mat4 view_{1.0f};
    glm::mat4 model_{1.0f};
    glm::mat4 mvp_{1.0f};

    // ==== 内部初始化步骤 ====
    bool create_instance();
    bool create_surface();
    bool pick_physical_device();
    bool create_logical_device();
    bool create_swapchain();
    bool create_image_views();
    bool create_render_pass();
    bool create_graphics_pipeline();
    bool create_framebuffers();
    bool create_command_pool();
    bool allocate_command_buffers();
    bool create_sync_objects();

    bool create_vertex_buffer(size_t size_bytes);
    bool create_index_buffer(size_t size_bytes);

    void record_command_buffers();

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    void cleanup_swapchain();
    void cleanup_vertex_index_buffers();

    // 辅助
    VkShaderModule create_shader_module(const std::vector<char>& code);
};
