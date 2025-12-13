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

    // ImGui / UI integration helpers
    VkInstance       vk_instance() const         { return instance_; }
    VkPhysicalDevice vk_physical_device() const  { return physical_device_; }
    VkDevice         vk_device() const           { return device_; }
    VkQueue          vk_graphics_queue() const   { return graphics_queue_; }
    uint32_t         graphics_queue_family_index() const { return graphics_queue_family_index_; }
    VkRenderPass     vk_render_pass() const      { return render_pass_; }
    VkCommandPool    vk_command_pool() const     { return command_pool_; }
    uint32_t         swapchain_image_count() const {
        return static_cast<uint32_t>(swapchain_images_.size());
    }

    bool init(GLFWwindow* window, uint32_t width, uint32_t height);
    void update_mesh(const std::vector<Vertex>& vertices,
                     const std::vector<u32>& indices);
    void update_box_mesh(const Vec3& center, const Vec3& half_extent);
    void update_ground_mesh(float y, float half_extent);
    void set_cloth_material(int index);
    void draw_frame();
    void wait_idle();
    void set_shadow_params(const glm::vec3& lightDir,
                           const glm::vec3& boxCenter,
                           float boxRadius,
                           const glm::vec3& clothCenter,
                           const glm::vec2& clothSize);

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

    std::vector<VkSemaphore> image_available_semaphores_;
    std::vector<VkSemaphore> render_finished_semaphores_;
    std::vector<VkFence>     in_flight_fences_;
    size_t current_frame_{0};
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

    // vertex/indices buffers
    VkBuffer       vertex_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory vertex_buffer_memory_{VK_NULL_HANDLE};
    size_t         vertex_buffer_size_{0};

    VkBuffer       index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory index_buffer_memory_{VK_NULL_HANDLE};
    size_t         index_count_{0};

    size_t index_buffer_size_{0};

    // materials and textures
    static constexpr int MAX_TEXTURES = 5;
    static constexpr int CLOTH_TEX_COUNT = 3;
    static constexpr int TEX_INDEX_CUBE   = 3;
    static constexpr int TEX_INDEX_GROUND = 4;

    VkImage        cloth_images_[MAX_TEXTURES]{};
    VkDeviceMemory cloth_image_memory_[MAX_TEXTURES]{};
    VkImageView    cloth_image_views_[MAX_TEXTURES]{};
    VkSampler      cloth_sampler_{VK_NULL_HANDLE};

    VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
    VkDescriptorPool      descriptor_pool_{VK_NULL_HANDLE};
    VkDescriptorSet       descriptor_set_{VK_NULL_HANDLE};

    int current_material_{0};


    // depth buffer
    VkImage        depth_image_{VK_NULL_HANDLE};
    VkDeviceMemory depth_image_memory_{VK_NULL_HANDLE};
    VkImageView    depth_image_view_{VK_NULL_HANDLE};
    VkFormat       depth_format_{VK_FORMAT_D32_SFLOAT};

    // cube buffer
    VkBuffer       cube_vertex_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory cube_vertex_memory_{VK_NULL_HANDLE};
    VkDeviceSize   cube_vertex_buffer_size_{0};

    VkBuffer       cube_index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory cube_index_memory_{VK_NULL_HANDLE};
    u32            cube_index_count_{0};

    // ground buffer
    VkBuffer       ground_vertex_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory ground_vertex_memory_{VK_NULL_HANDLE};
    VkDeviceSize   ground_vertex_buffer_size_{0};

    VkBuffer       ground_index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory ground_index_memory_{VK_NULL_HANDLE};
    u32            ground_index_count_{0};

    // Shadows
    glm::vec3 light_dir_        = glm::vec3(0.3f, 0.6f, -0.3f);
    glm::vec3 shadow_box_center_  = glm::vec3(0.0f);
    float     shadow_box_radius_  = 0.5f;
    glm::vec3 shadow_cloth_center_ = glm::vec3(0.0f);
    glm::vec2 shadow_cloth_size_   = glm::vec2(1.0f, 1.0f);

    // helpers
    uint32_t graphics_queue_family_index_{0};

    // MVP
    glm::mat4 proj_{1.0f};
    glm::mat4 view_{1.0f};
    glm::mat4 model_{1.0f};
    glm::mat4 mvp_{1.0f};

    // iniatilization
    bool create_instance();
    bool create_surface();
    bool pick_physical_device();
    bool create_logical_device();
    bool create_swapchain();
    bool create_image_views();
    bool create_depth_resources();
    bool create_render_pass();
    bool create_graphics_pipeline();
    bool create_framebuffers();
    bool create_command_pool();
    bool allocate_command_buffers();
    bool create_sync_objects();

    bool create_vertex_buffer(size_t size_bytes);
    bool create_index_buffer(size_t size_bytes);

    void record_command_buffer(uint32_t imageIndex);

    uint32_t find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties);
    void cleanup_swapchain();
    void cleanup_vertex_index_buffers();

    // helper
    VkShaderModule create_shader_module(const std::vector<char>& code);
    
    // UI
    VkDescriptorPool imgui_descriptor_pool_ = VK_NULL_HANDLE;
    bool             imgui_initialized_     = false;

    // materials and texture related
    bool create_descriptor_set_layout();
    bool create_descriptor_pool();
    bool create_descriptor_set();
    bool create_cloth_textures();
    bool create_cloth_sampler();

    VkCommandBuffer begin_single_time_commands();
    void end_single_time_commands(VkCommandBuffer cmd);
};
