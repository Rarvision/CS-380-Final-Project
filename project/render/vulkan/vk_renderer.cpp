#include "vk_renderer.hpp"

#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>
#include <array>
#include <optional>
#include <cstring>
#include <fstream>
#include <iostream>
#include <algorithm>

#include <glm/glm.hpp>
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include "../../third_party/stb_image.h"
#include "../../third_party/imgui/imgui.h"
#include "../../third_party/imgui/imgui_impl_vulkan.h"

static bool g_imgui_fonts_uploaded = false; 

struct PushConstants {
    glm::mat4 mvp;

    // params0: xyz = lightDir（世界坐标）, w = materialIndex
    glm::vec4 params0;

    // params1: xyz = boxCenter（世界坐标）, w = boxRadius（在 xz 平面上的半径）
    glm::vec4 params1;

    // params2: xyz = clothCenter（世界坐标）, w = 1.0
    glm::vec4 params2;

    // params3: xy = clothSize（xz 尺寸）, zw 预留
    glm::vec4 params3;
};

static_assert(sizeof(PushConstants) <= 128, "PushConstants too big");
static constexpr uint32_t PUSH_CONSTANT_SIZE = sizeof(PushConstants);

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics_family;
    std::optional<uint32_t> present_family;

    bool is_complete() const {
        return graphics_family.has_value() && present_family.has_value();
    }
};

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR        capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   present_modes;
};

// =================== 工具函数 ===================

static std::vector<char> read_file(const std::string& filename)
{
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file) throw std::runtime_error("failed to open file: " + filename);

    size_t file_size = (size_t)file.tellg();
    std::vector<char> buffer(file_size);
    file.seekg(0);
    file.read(buffer.data(), file_size);
    file.close();
    return buffer;
}

static QueueFamilyIndices find_queue_families(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    QueueFamilyIndices indices;

    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        const auto& f = families[i];
        if (f.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics_family = i;
        }

        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
        if (present_support) {
            indices.present_family = i;
        }

        if (indices.is_complete()) break;
    }

    return indices;
}

static SwapChainSupportDetails query_swapchain_support(VkPhysicalDevice device, VkSurfaceKHR surface)
{
    SwapChainSupportDetails details{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, nullptr);
    if (count) {
        details.formats.resize(count);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &count, details.formats.data());
    }

    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
    if (count) {
        details.present_modes.resize(count);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, details.present_modes.data());
    }

    return details;
}

static VkSurfaceFormatKHR choose_swap_surface_format(const std::vector<VkSurfaceFormatKHR>& formats)
{
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats[0];
}

static VkPresentModeKHR choose_swap_present_mode(const std::vector<VkPresentModeKHR>& modes)
{
    for (const auto& m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

static VkExtent2D choose_swap_extent(const VkSurfaceCapabilitiesKHR& caps, uint32_t width, uint32_t height)
{
    if (caps.currentExtent.width != UINT32_MAX) {
        return caps.currentExtent;
    } else {
        VkExtent2D actual{width, height};
        actual.width  = std::clamp(actual.width,  caps.minImageExtent.width,  caps.maxImageExtent.width);
        actual.height = std::clamp(actual.height, caps.minImageExtent.height, caps.maxImageExtent.height);
        return actual;
    }
}

// =================== VulkanRenderer 实现 ===================

VulkanRenderer::~VulkanRenderer()
{
    // std::cerr << "[VK] ~VulkanRenderer begin\n";

    if (device_ != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device_);
    }

    cleanup_vertex_index_buffers();

    cleanup_swapchain();

    for (size_t i = 0; i < image_available_semaphores_.size(); ++i) {
        if (image_available_semaphores_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
        }
    }
    for (size_t i = 0; i < render_finished_semaphores_.size(); ++i) {
        if (render_finished_semaphores_[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
        }
    }
    for (size_t i = 0; i < in_flight_fences_.size(); ++i) {
        if (in_flight_fences_[i] != VK_NULL_HANDLE) {
            vkDestroyFence(device_, in_flight_fences_[i], nullptr);
        }
    }
    image_available_semaphores_.clear();
    render_finished_semaphores_.clear();
    in_flight_fences_.clear();

    if (graphics_pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
        graphics_pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        render_pass_ = VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }
    command_buffers_.clear();

    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }

    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }

    // std::cerr << "[VK] ~VulkanRenderer end\n";

}


bool VulkanRenderer::init(GLFWwindow* window, uint32_t width, uint32_t height)
{
    window_ = window;

    if (!create_instance())        return false;
    if (!create_surface())         return false;
    if (!pick_physical_device())   return false;
    if (!create_logical_device())  return false;
    if (!create_swapchain())       return false;
    if (!create_image_views())     return false;
    if (!create_depth_resources()) return false;

    if (!create_descriptor_set_layout()) return false;

    if (!create_render_pass())     return false;
    if (!create_graphics_pipeline()) return false;
    if (!create_framebuffers())    return false;
    if (!create_command_pool())    return false;
    if (!allocate_command_buffers()) return false;
    if (!create_sync_objects())    return false;

    // 先创建一个空的顶点/索引缓冲（后面 update_mesh 会重建）
    create_vertex_buffer(sizeof(Vertex) * 1);
    create_index_buffer(sizeof(uint32_t) * 1);

    // ★ 创建纹理资源 + descriptor
    if (!create_cloth_textures())   return false;
    if (!create_cloth_sampler())    return false;
    if (!create_descriptor_pool())  return false;
    if (!create_descriptor_set())   return false;

    // ====== 计算一个简单 MVP 矩阵 ======
    float aspect = static_cast<float>(swapchain_extent_.width) /
                   static_cast<float>(swapchain_extent_.height);

    proj_ = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 10.0f);
    // Vulkan NDC 的 Y 是反的，通常要翻一下glm::mat4 proj
    proj_[1][1] *= -1.0f;

    view_ = glm::lookAt(
        glm::vec3(0.0f, 0.2f, -5.0f),  // 相机位置：稍微在前面、略微偏上
        glm::vec3(0.0f, 0.0f, 0.0f),  // 看向原点
        glm::vec3(0.0f, 1.0f, 0.0f)   // 世界向上方向
    );

    model_ = glm::mat4(1.0f); // 后面要旋转布的话就改这里
    mvp_ = proj_ * view_ * model_;

    return true;
}

void VulkanRenderer::wait_idle()
{
    if (device_) vkDeviceWaitIdle(device_);
}

// ========== 创建 Instance / Surface / Device / Swapchain ==========

bool VulkanRenderer::create_instance()
{
    VkApplicationInfo app{};
    app.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName   = "ClothSim";
    app.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app.pEngineName        = "NoEngine";
    app.engineVersion      = VK_MAKE_VERSION(1, 0, 0);
    app.apiVersion         = VK_API_VERSION_1_1;

    uint32_t glfwExtCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);

    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    info.enabledExtensionCount   = glfwExtCount;
    info.ppEnabledExtensionNames = glfwExts;
    info.enabledLayerCount       = 0;

    if (vkCreateInstance(&info, nullptr, &instance_) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan instance\n";
        return false;
    }
    return true;
}


bool VulkanRenderer::create_surface()
{
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS) {
        std::cerr << "Failed to create window surface\n";
        return false;
    }
    return true;
}

bool VulkanRenderer::pick_physical_device()
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) {
        std::cerr << "No Vulkan physical devices\n";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    for (auto dev : devices) {
        QueueFamilyIndices indices = find_queue_families(dev, surface_);
        bool extensions_ok = false;

        // 检查 swapchain 扩展
        uint32_t extCount = 0;
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateDeviceExtensionProperties(dev, nullptr, &extCount, exts.data());

        const char* required = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
        for (const auto& e : exts) {
            if (std::strcmp(e.extensionName, required) == 0) {
                extensions_ok = true;
                break;
            }
        }

        if (!indices.is_complete() || !extensions_ok) continue;

        SwapChainSupportDetails sc = query_swapchain_support(dev, surface_);
        if (sc.formats.empty() || sc.present_modes.empty()) continue;

        physical_device_ = dev;
        return true;
    }

    std::cerr << "Failed to find suitable GPU\n";
    return false;
}

bool VulkanRenderer::create_logical_device()
{
    QueueFamilyIndices indices = find_queue_families(physical_device_, surface_);

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    std::vector<uint32_t> uniqueFamilies;
    uniqueFamilies.push_back(indices.graphics_family.value());
    if (indices.present_family != indices.graphics_family) {
        uniqueFamilies.push_back(indices.present_family.value());
    }

    float queuePriority = 1.0f;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo q{};
        q.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        q.queueFamilyIndex = family;
        q.queueCount       = 1;
        q.pQueuePriorities = &queuePriority;
        queueInfos.push_back(q);
    }

    VkPhysicalDeviceFeatures features{};
    features.samplerAnisotropy = VK_FALSE;

    const char* deviceExtensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

    VkDeviceCreateInfo info{};
    info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size());
    info.pQueueCreateInfos       = queueInfos.data();
    info.pEnabledFeatures        = &features;
    info.enabledExtensionCount   = 1;
    info.ppEnabledExtensionNames = deviceExtensions;
    info.enabledLayerCount       = 0;

    if (vkCreateDevice(physical_device_, &info, nullptr, &device_) != VK_SUCCESS) {
        std::cerr << "Failed to create logical device\n";
        return false;
    }

    graphics_queue_family_index_ = indices.graphics_family.value();

    vkGetDeviceQueue(device_, indices.graphics_family.value(), 0, &graphics_queue_);
    vkGetDeviceQueue(device_, indices.present_family.value(), 0, &present_queue_);
    return true;
}

bool VulkanRenderer::create_swapchain()
{
    SwapChainSupportDetails support = query_swapchain_support(physical_device_, surface_);
    if (support.formats.empty() || support.present_modes.empty()) {
        std::cerr << "Swapchain support incomplete\n";
        return false;
    }

    VkSurfaceFormatKHR surfaceFormat = choose_swap_surface_format(support.formats);
    VkPresentModeKHR   presentMode   = choose_swap_present_mode(support.present_modes);
    int w, h;
    glfwGetFramebufferSize(window_, &w, &h);
    VkExtent2D extent = choose_swap_extent(support.capabilities, (uint32_t)w, (uint32_t)h);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 &&
        imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR info{};
    info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface          = surface_;
    info.minImageCount    = imageCount;
    info.imageFormat      = surfaceFormat.format;
    info.imageColorSpace  = surfaceFormat.colorSpace;
    info.imageExtent      = extent;
    info.imageArrayLayers = 1;
    info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = find_queue_families(physical_device_, surface_);
    uint32_t queueFamilyIndices[] = { indices.graphics_family.value(),
                                      indices.present_family.value() };

    if (indices.graphics_family != indices.present_family) {
        info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        info.queueFamilyIndexCount = 2;
        info.pQueueFamilyIndices   = queueFamilyIndices;
    } else {
        info.imageSharingMode      = VK_SHARING_MODE_EXCLUSIVE;
    }

    info.preTransform   = support.capabilities.currentTransform;
    info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    info.presentMode    = presentMode;
    info.clipped        = VK_TRUE;
    info.oldSwapchain   = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_, &info, nullptr, &swapchain_) != VK_SUCCESS) {
        std::cerr << "Failed to create swapchain\n";
        return false;
    }

    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, nullptr);
    swapchain_images_.resize(imageCount);
    vkGetSwapchainImagesKHR(device_, swapchain_, &imageCount, swapchain_images_.data());

    swapchain_image_format_ = surfaceFormat.format;
    swapchain_extent_       = extent;

    return true;
}

bool VulkanRenderer::create_image_views()
{
    swapchain_image_views_.resize(swapchain_images_.size());

    for (size_t i = 0; i < swapchain_images_.size(); ++i) {
        VkImageViewCreateInfo info{};
        info.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        info.image    = swapchain_images_[i];
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format   = swapchain_image_format_;
        info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        info.subresourceRange.baseMipLevel   = 0;
        info.subresourceRange.levelCount     = 1;
        info.subresourceRange.baseArrayLayer = 0;
        info.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(device_, &info, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create image view\n";
            return false;
        }
    }
    return true;
}

bool VulkanRenderer::create_render_pass()
{
    VkAttachmentDescription color{};
    color.format         = swapchain_image_format_;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format         = depth_format_;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription attachments[2] = { color, depth };

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1;
    subpass.pColorAttachments       = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    VkRenderPassCreateInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 2;
    rp.pAttachments    = attachments;
    rp.subpassCount    = 1;
    rp.pSubpasses      = &subpass;

    if (vkCreateRenderPass(device_, &rp, nullptr, &render_pass_) != VK_SUCCESS) {
        std::cerr << "Failed to create render pass\n";
        return false;
    }
    return true;
}

VkShaderModule VulkanRenderer::create_shader_module(const std::vector<char>& code)
{
    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = code.size();
    info.pCode    = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module;
    if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS) {
        throw std::runtime_error("failed to create shader module");
    }
    return module;
}

bool VulkanRenderer::create_graphics_pipeline()
{
    auto vertCode = read_file("../assets/shaders/cloth.vert.spv");
    auto fragCode = read_file("../assets/shaders/cloth.frag.spv");

    VkShaderModule vertModule = create_shader_module(vertCode);
    VkShaderModule fragModule = create_shader_module(fragCode);

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vertModule;
    vertStage.pName  = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = fragModule;
    fragStage.pName  = "main";

    VkPipelineShaderStageCreateInfo stages[] = { vertStage, fragStage };

    auto bindingDesc = Vertex::binding_description();
    auto attrDescs   = Vertex::attribute_descriptions();

    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount   = 1;
    vertexInput.pVertexBindingDescriptions      = &bindingDesc;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrDescs.size());
    vertexInput.pVertexAttributeDescriptions    = attrDescs.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = static_cast<float>(swapchain_extent_.width);
    viewport.height   = static_cast<float>(swapchain_extent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapchain_extent_;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports    = &viewport;
    viewportState.scissorCount  = 1;
    viewportState.pScissors     = &scissor;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.depthClampEnable        = VK_FALSE;
    raster.rasterizerDiscardEnable = VK_FALSE;
    raster.polygonMode             = VK_POLYGON_MODE_FILL;
    raster.lineWidth               = 1.0f;
    raster.cullMode                = VK_CULL_MODE_NONE;
    raster.frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.depthBiasEnable         = VK_TRUE;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable  = VK_FALSE;

    VkDynamicState dynamicStates[] = {
        VK_DYNAMIC_STATE_DEPTH_BIAS
    };

    VkPipelineDynamicStateCreateInfo dynamicInfo{};
    dynamicInfo.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicInfo.dynamicStateCount = 1;
    dynamicInfo.pDynamicStates    = dynamicStates;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.colorBlendOp        = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachment.alphaBlendOp        = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.logicOpEnable   = VK_FALSE;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &colorBlendAttachment;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = PUSH_CONSTANT_SIZE;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &descriptor_set_layout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;

    if (vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable       = VK_TRUE;
    depthStencil.depthWriteEnable      = VK_TRUE;
    depthStencil.depthCompareOp        = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable     = VK_FALSE;

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = stages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlend;
    pipelineInfo.pDynamicState       = &dynamicInfo;
    pipelineInfo.layout              = pipeline_layout_;
    pipelineInfo.renderPass          = render_pass_;
    pipelineInfo.subpass             = 0;

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphics_pipeline_) != VK_SUCCESS) {
        std::cerr << "Failed to create graphics pipeline\n";
        return false;
    }

    vkDestroyShaderModule(device_, vertModule, nullptr);
    vkDestroyShaderModule(device_, fragModule, nullptr);
    return true;
}

bool VulkanRenderer::create_framebuffers()
{
    framebuffers_.resize(swapchain_image_views_.size());

    for (size_t i = 0; i < swapchain_image_views_.size(); ++i) {
        VkImageView attachments[] = { swapchain_image_views_[i], depth_image_view_ };

        VkFramebufferCreateInfo info{};
        info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass      = render_pass_;
        info.attachmentCount = 2;
        info.pAttachments    = attachments;
        info.width           = swapchain_extent_.width;
        info.height          = swapchain_extent_.height;
        info.layers          = 1;

        if (vkCreateFramebuffer(device_, &info, nullptr, &framebuffers_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create framebuffer\n";
            return false;
        }
    }
    return true;
}

bool VulkanRenderer::create_command_pool()
{
    QueueFamilyIndices indices = find_queue_families(physical_device_, surface_);

    VkCommandPoolCreateInfo info{};
    info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.queueFamilyIndex = indices.graphics_family.value();
    info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (vkCreateCommandPool(device_, &info, nullptr, &command_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create command pool\n";
        return false;
    }
    return true;
}

bool VulkanRenderer::allocate_command_buffers()
{
    command_buffers_.resize(framebuffers_.size());

    VkCommandBufferAllocateInfo info{};
    info.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    info.commandPool        = command_pool_;
    info.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = static_cast<uint32_t>(command_buffers_.size());

    if (vkAllocateCommandBuffers(device_, &info, command_buffers_.data()) != VK_SUCCESS) {
        std::cerr << "Failed to allocate command buffers\n";
        return false;
    }
    return true;
}

bool VulkanRenderer::create_sync_objects()
{
    image_available_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    render_finished_semaphores_.resize(MAX_FRAMES_IN_FLIGHT);
    in_flight_fences_.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (vkCreateSemaphore(device_, &semInfo, nullptr, &image_available_semaphores_[i]) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &semInfo, nullptr, &render_finished_semaphores_[i]) != VK_SUCCESS ||
            vkCreateFence(device_, &fenceInfo, nullptr, &in_flight_fences_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create sync objects\n";
            return false;
        }
    }
    return true;
}

// =================== 顶点/索引缓冲 ===================

bool VulkanRenderer::create_vertex_buffer(size_t size_bytes)
{
    vertex_buffer_size_ = size_bytes;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size  = size_bytes;
    info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &info, nullptr, &vertex_buffer_) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, vertex_buffer_, &memReq);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = memReq.size;
    alloc.memoryTypeIndex = find_memory_type(
        memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device_, &alloc, nullptr, &vertex_buffer_memory_) != VK_SUCCESS) {
        return false;
    }

    vkBindBufferMemory(device_, vertex_buffer_, vertex_buffer_memory_, 0);
    return true;
}


bool VulkanRenderer::create_index_buffer(size_t size_bytes)
{
    index_buffer_size_ = size_bytes;

    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size  = size_bytes;
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &info, nullptr, &index_buffer_) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, index_buffer_, &memReq);

    VkMemoryAllocateInfo alloc{};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = memReq.size;
    alloc.memoryTypeIndex = find_memory_type(
        memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device_, &alloc, nullptr, &index_buffer_memory_) != VK_SUCCESS) {
        return false;
    }

    vkBindBufferMemory(device_, index_buffer_, index_buffer_memory_, 0);
    return true;
}

bool VulkanRenderer::create_depth_resources()
{
    depth_format_ = VK_FORMAT_D32_SFLOAT;
    VkImageCreateInfo img{};
    img.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    img.imageType = VK_IMAGE_TYPE_2D;
    img.extent.width  = swapchain_extent_.width;
    img.extent.height = swapchain_extent_.height;
    img.extent.depth  = 1;
    img.mipLevels     = 1;
    img.arrayLayers   = 1;
    img.format        = depth_format_;
    img.tiling        = VK_IMAGE_TILING_OPTIMAL;
    img.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    img.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    img.samples       = VK_SAMPLE_COUNT_1_BIT;
    img.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device_, &img, nullptr, &depth_image_) != VK_SUCCESS) {
        std::cerr << "Failed to create depth image\n";
        return false;
    }

    VkMemoryRequirements memReq;
    vkGetImageMemoryRequirements(device_, depth_image_, &memReq);

    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = memReq.size;
    alloc.memoryTypeIndex = find_memory_type(
        memReq.memoryTypeBits,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    if (vkAllocateMemory(device_, &alloc, nullptr, &depth_image_memory_) != VK_SUCCESS) {
        std::cerr << "Failed to allocate depth image memory\n";
        return false;
    }

    vkBindImageMemory(device_, depth_image_, depth_image_memory_, 0);

    // 创建 image view
    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image    = depth_image_;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format   = depth_format_;
    viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    viewInfo.subresourceRange.baseMipLevel   = 0;
    viewInfo.subresourceRange.levelCount     = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount     = 1;

    if (vkCreateImageView(device_, &viewInfo, nullptr, &depth_image_view_) != VK_SUCCESS) {
        std::cerr << "Failed to create depth image view\n";
        return false;
    }

    return true;
}


void VulkanRenderer::update_mesh(const std::vector<Vertex>& vertices,
                                 const std::vector<u32>& indices)
{
//    std::cerr << "[VK] update_mesh: vertices=" << vertices.size()
//              << " indices=" << indices.size() << std::endl;

    size_t vbytes = vertices.size() * sizeof(Vertex);
    size_t ibytes = indices.size()  * sizeof(uint32_t);

    if (vertex_buffer_ == VK_NULL_HANDLE || vbytes > vertex_buffer_size_) {
        if (vertex_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, vertex_buffer_, nullptr);
            vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
            vertex_buffer_ = VK_NULL_HANDLE;
            vertex_buffer_memory_ = VK_NULL_HANDLE;
            vertex_buffer_size_ = 0;
        }
        if (!create_vertex_buffer(vbytes)) {
            throw std::runtime_error("Failed to create vertex buffer");
        }
    }

    if (index_buffer_ == VK_NULL_HANDLE || ibytes > index_buffer_size_) {
        if (index_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, index_buffer_, nullptr);
            vkFreeMemory(device_, index_buffer_memory_, nullptr);
            index_buffer_ = VK_NULL_HANDLE;
            index_buffer_memory_ = VK_NULL_HANDLE;
            index_buffer_size_ = 0;
        }
        if (!create_index_buffer(ibytes)) {
            throw std::runtime_error("Failed to create index buffer");
        }
    }

    // CPU → GPU 拷贝
    void* data = nullptr;
    vkMapMemory(device_, vertex_buffer_memory_, 0, vbytes, 0, &data);
    std::memcpy(data, vertices.data(), vbytes);
    vkUnmapMemory(device_, vertex_buffer_memory_);

    vkMapMemory(device_, index_buffer_memory_, 0, ibytes, 0, &data);
    std::memcpy(data, indices.data(), ibytes);
    vkUnmapMemory(device_, index_buffer_memory_);

    index_count_ = indices.size();
}

void VulkanRenderer::update_box_mesh(const Vec3& center, const Vec3& half_extent)
{
    const float cx = center.x, cy = center.y, cz = center.z;
    const float hx = half_extent.x, hy = half_extent.y, hz = half_extent.z;

    std::vector<Vertex> verts;
    verts.reserve(24);
    std::vector<u32> indices;
    indices.reserve(36);

    auto push_face = [&](Vec3 v0, Vec3 v1, Vec3 v2, Vec3 v3, Vec3 n,
                         bool flip_uv_y)
    {
        u32 start = static_cast<u32>(verts.size());

        Vertex a{}, b{}, c{}, d{};
        a.pos = v0; b.pos = v1; c.pos = v2; d.pos = v3;
        a.normal = b.normal = c.normal = d.normal = n;

        // 统一 uv: (0,0) (1,0) (1,1) (0,1)
        a.uv = glm::vec2(0.0f, 0.0f);
        b.uv = glm::vec2(1.0f, 0.0f);
        c.uv = glm::vec2(1.0f, 1.0f);
        d.uv = glm::vec2(0.0f, 1.0f);

        if (flip_uv_y) {
            a.uv.y = 1.0f - a.uv.y;
            b.uv.y = 1.0f - b.uv.y;
            c.uv.y = 1.0f - c.uv.y;
            d.uv.y = 1.0f - d.uv.y;
        }

        verts.push_back(a);
        verts.push_back(b);
        verts.push_back(c);
        verts.push_back(d);

        // 两个三角： (0,1,2) (2,3,0)
        indices.push_back(start + 0);
        indices.push_back(start + 1);
        indices.push_back(start + 2);
        indices.push_back(start + 2);
        indices.push_back(start + 3);
        indices.push_back(start + 0);
    };

    // front (-Z)
    push_face(
        Vec3(cx - hx, cy - hy, cz - hz),
        Vec3(cx + hx, cy - hy, cz - hz),
        Vec3(cx + hx, cy + hy, cz - hz),
        Vec3(cx - hx, cy + hy, cz - hz),
        Vec3(0.0f, 0.0f, -1.0f),
        false
    );

    // back (+Z)
    push_face(
        Vec3(cx + hx, cy - hy, cz + hz),
        Vec3(cx - hx, cy - hy, cz + hz),
        Vec3(cx - hx, cy + hy, cz + hz),
        Vec3(cx + hx, cy + hy, cz + hz),
        Vec3(0.0f, 0.0f, 1.0f),
        false
    );

    // left (-X)
    push_face(
        Vec3(cx - hx, cy - hy, cz + hz),
        Vec3(cx - hx, cy - hy, cz - hz),
        Vec3(cx - hx, cy + hy, cz - hz),
        Vec3(cx - hx, cy + hy, cz + hz),
        Vec3(-1.0f, 0.0f, 0.0f),
        false
    );

    // right (+X)
    push_face(
        Vec3(cx + hx, cy - hy, cz - hz),
        Vec3(cx + hx, cy - hy, cz + hz),
        Vec3(cx + hx, cy + hy, cz + hz),
        Vec3(cx + hx, cy + hy, cz - hz),
        Vec3(1.0f, 0.0f, 0.0f),
        false
    );

    // bottom (-Y)
    push_face(
        Vec3(cx - hx, cy - hy, cz + hz),
        Vec3(cx + hx, cy - hy, cz + hz),
        Vec3(cx + hx, cy - hy, cz - hz),
        Vec3(cx - hx, cy - hy, cz - hz),
        Vec3(0.0f, -1.0f, 0.0f),
        true   // 这里如果方向颠倒，可以 flip_uv_y 再试
    );

    // top (+Y)
    push_face(
        Vec3(cx - hx, cy + hy, cz - hz),
        Vec3(cx + hx, cy + hy, cz - hz),
        Vec3(cx + hx, cy + hy, cz + hz),
        Vec3(cx - hx, cy + hy, cz + hz),
        Vec3(0.0f, 1.0f, 0.0f),
        false
    );

    // 后面的 buffer 分配和拷贝逻辑不变，只是改用 verts / indices

    VkDeviceSize vbytes = sizeof(Vertex) * verts.size();
    VkDeviceSize ibytes = sizeof(u32)    * indices.size();

    // 顶点缓冲：同你现在的实现
    if (cube_vertex_buffer_ == VK_NULL_HANDLE || vbytes > cube_vertex_buffer_size_) {
        if (cube_vertex_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, cube_vertex_buffer_, nullptr);
            vkFreeMemory(device_, cube_vertex_memory_, nullptr);
        }
        cube_vertex_buffer_size_ = vbytes;
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = vbytes;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device_, &bufferInfo, nullptr, &cube_vertex_buffer_);
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, cube_vertex_buffer_, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device_, &allocInfo, nullptr, &cube_vertex_memory_);
        vkBindBufferMemory(device_, cube_vertex_buffer_, cube_vertex_memory_, 0);
    }

    if (cube_index_buffer_ == VK_NULL_HANDLE || ibytes > cube_index_count_ * sizeof(u32)) {
        if (cube_index_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, cube_index_buffer_, nullptr);
            vkFreeMemory(device_, cube_index_memory_, nullptr);
        }
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = ibytes;
        bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device_, &bufferInfo, nullptr, &cube_index_buffer_);
        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, cube_index_buffer_, &memReq);
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        vkAllocateMemory(device_, &allocInfo, nullptr, &cube_index_memory_);
        vkBindBufferMemory(device_, cube_index_buffer_, cube_index_memory_, 0);
    }

    void* data = nullptr;
    vkMapMemory(device_, cube_vertex_memory_, 0, vbytes, 0, &data);
    std::memcpy(data, verts.data(), vbytes);
    vkUnmapMemory(device_, cube_vertex_memory_);

    vkMapMemory(device_, cube_index_memory_, 0, ibytes, 0, &data);
    std::memcpy(data, indices.data(), ibytes);
    vkUnmapMemory(device_, cube_index_memory_);

    cube_index_count_ = static_cast<u32>(indices.size());
}

void VulkanRenderer::update_ground_mesh(float y, float half_extent)
{
    // 做一个大方形地板：[-h, h] x [-h, h]，位于 y 高度
    std::vector<Vertex> verts(4);
    float h = half_extent;

    verts[0].pos = Vec3(-h, y, -h);
    verts[1].pos = Vec3( h, y, -h);
    verts[2].pos = Vec3( h, y,  h);
    verts[3].pos = Vec3(-h, y,  h);

    for (auto& v : verts) {
        v.normal = Vec3(0.0f, 1.0f, 0.0f);
    }

    float tile = 1.0f;
    verts[0].uv = glm::vec2(0.0f, 0.0f);
    verts[1].uv = glm::vec2(tile, 0.0f);
    verts[2].uv = glm::vec2(tile, tile);
    verts[3].uv = glm::vec2(0.0f, tile);

    std::vector<u32> indices = {
        0, 1, 2,
        2, 3, 0
    };

    VkDeviceSize vbytes = sizeof(Vertex) * verts.size();
    VkDeviceSize ibytes = sizeof(u32)    * indices.size();

    // --- 顶点缓冲 ---
    if (ground_vertex_buffer_ == VK_NULL_HANDLE || vbytes > ground_vertex_buffer_size_) {
        if (ground_vertex_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, ground_vertex_buffer_, nullptr);
            vkFreeMemory(device_, ground_vertex_memory_, nullptr);
            ground_vertex_buffer_ = VK_NULL_HANDLE;
            ground_vertex_memory_ = VK_NULL_HANDLE;
            ground_vertex_buffer_size_ = 0;
        }

        ground_vertex_buffer_size_ = vbytes;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = vbytes;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &ground_vertex_buffer_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create ground vertex buffer");
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, ground_vertex_buffer_, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &ground_vertex_memory_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate ground vertex buffer memory");
        }

        vkBindBufferMemory(device_, ground_vertex_buffer_, ground_vertex_memory_, 0);
    }

    // --- 索引缓冲 ---
    if (ground_index_buffer_ == VK_NULL_HANDLE || ibytes > ground_index_count_ * sizeof(u32)) {
        if (ground_index_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, ground_index_buffer_, nullptr);
            vkFreeMemory(device_, ground_index_memory_, nullptr);
            ground_index_buffer_ = VK_NULL_HANDLE;
            ground_index_memory_ = VK_NULL_HANDLE;
            ground_index_count_  = 0;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = ibytes;
        bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &ground_index_buffer_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create ground index buffer");
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, ground_index_buffer_, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &ground_index_memory_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate ground index buffer memory");
        }

        vkBindBufferMemory(device_, ground_index_buffer_, ground_index_memory_, 0);
    }

    // 拷贝数据
    void* data = nullptr;
    vkMapMemory(device_, ground_vertex_memory_, 0, vbytes, 0, &data);
    std::memcpy(data, verts.data(), vbytes);
    vkUnmapMemory(device_, ground_vertex_memory_);

    vkMapMemory(device_, ground_index_memory_, 0, ibytes, 0, &data);
    std::memcpy(data, indices.data(), ibytes);
    vkUnmapMemory(device_, ground_index_memory_);

    ground_index_count_ = static_cast<u32>(indices.size());
}

// =================== 命令缓冲 & draw ===================
// void VulkanRenderer::record_command_buffer(uint32_t imageIndex)
// {
//     VkCommandBuffer cmd = command_buffers_[imageIndex];

//     VkCommandBufferBeginInfo beginInfo{};
//     beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

//     if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
//         throw std::runtime_error("failed to begin recording command buffer");
//     }

//     VkRenderPassBeginInfo rpBegin{};
//     rpBegin.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
//     rpBegin.renderPass  = render_pass_;
//     rpBegin.framebuffer = framebuffers_[imageIndex];
//     rpBegin.renderArea.offset = {0, 0};
//     rpBegin.renderArea.extent = swapchain_extent_;

//     VkClearValue clearValues[2];
//     clearValues[0].color        = {{0.02f, 0.02f, 0.05f, 1.0f}};
//     clearValues[1].depthStencil = {1.0f, 0};

//     rpBegin.clearValueCount = 2;
//     rpBegin.pClearValues    = clearValues;

//     vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

//     vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

//     // 绑定纹理 descriptor set（set = 0）
//     vkCmdBindDescriptorSets(
//         cmd,
//         VK_PIPELINE_BIND_POINT_GRAPHICS,
//         pipeline_layout_,
//         0,
//         1,
//         &descriptor_set_,
//         0,
//         nullptr
//     );

//     // ==== 填充公共 push 常量 ====
//     PushConstants pc{};
//     pc.mvp = mvp_;

//     // params0.xyz = lightDir, w = materialIndex（先填 cloth 当前材质）
//     pc.params0 = glm::vec4(light_dir_, float(current_material_));

//     // params1.xyz = boxCenter, w = boxRadius
//     pc.params1 = glm::vec4(shadow_box_center_, shadow_box_radius_);

//     // params2.xyz = clothCenter
//     pc.params2 = glm::vec4(shadow_cloth_center_, 1.0f);

//     // params3.xy = clothSize
//     pc.params3 = glm::vec4(shadow_cloth_size_, 0.0f, 0.0f);

//     // ===== 1) 地面：materialIndex = 4 =====
//     if (ground_index_count_ > 0) {
//         pc.params0.w = 4.0f;  // 地面材质 ID（在 shader 里用它区分）

//         vkCmdPushConstants(
//             cmd,
//             pipeline_layout_,
//             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
//             0,
//             PUSH_CONSTANT_SIZE,
//             &pc
//         );

//         VkBuffer vb[]       = { ground_vertex_buffer_ };
//         VkDeviceSize offs[] = { 0 };
//         vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
//         vkCmdBindIndexBuffer(cmd, ground_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
//         vkCmdDrawIndexed(cmd, ground_index_count_, 1, 0, 0, 0);
//     }

//     // ===== 2) 方块：materialIndex = 3 =====
//     if (cube_index_count_ > 0) {
//         pc.params0.w = 3.0f;  // 方块材质 ID

//         vkCmdPushConstants(
//             cmd,
//             pipeline_layout_,
//             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
//             0,
//             PUSH_CONSTANT_SIZE,
//             &pc
//         );

//         VkBuffer vb[]       = { cube_vertex_buffer_ };
//         VkDeviceSize offs[] = { 0 };
//         vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
//         vkCmdBindIndexBuffer(cmd, cube_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
//         vkCmdDrawIndexed(cmd, cube_index_count_, 1, 0, 0, 0);
//     }

//     // ===== 3) 布：materialIndex = current_material_ (0/1/2) =====
//     if (index_count_ > 0) {
//         pc.params0.w = float(current_material_);

//         vkCmdPushConstants(
//             cmd,
//             pipeline_layout_,
//             VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
//             0,
//             PUSH_CONSTANT_SIZE,
//             &pc
//         );

//         VkBuffer vb[]       = { vertex_buffer_ };
//         VkDeviceSize offs[] = { 0 };
//         vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
//         vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
//         vkCmdDrawIndexed(cmd, static_cast<uint32_t>(index_count_), 1, 0, 0, 0);
//     }

//     // ===== 4) ImGui overlay =====
//     ImDrawData* draw_data = ImGui::GetDrawData();
//     if (draw_data && draw_data->CmdListsCount > 0) {
//         ImGui_ImplVulkan_RenderDrawData(draw_data, cmd, VK_NULL_HANDLE);
//     }

//     vkCmdEndRenderPass(cmd);

//     if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
//         throw std::runtime_error("failed to record command buffer");
//     }
// }
void VulkanRenderer::record_command_buffer(uint32_t imageIndex)
{
    VkCommandBuffer cmd = command_buffers_[imageIndex];

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer");
    }

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass  = render_pass_;
    rpBegin.framebuffer = framebuffers_[imageIndex];
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = swapchain_extent_;

    VkClearValue clearValues[2];
    clearValues[0].color        = {{0.02f, 0.02f, 0.05f, 1.0f}};
    clearValues[1].depthStencil = {1.0f, 0};

    rpBegin.clearValueCount = 2;
    rpBegin.pClearValues    = clearValues;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

    // 绑定纹理数组（set = 0, binding = 0）
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout_,
        0,
        1,
        &descriptor_set_,
        0,
        nullptr
    );

    // 公共 push 常量：这一帧所有物体共享
    PushConstants pc{};
    pc.mvp = mvp_;

    // 从 set_shadow_params 里填过来的数据
    pc.params0 = glm::vec4(light_dir_, 0.0f);                 // w 先占位，后面根据物体改
    pc.params1 = glm::vec4(shadow_box_center_, shadow_box_radius_);
    pc.params2 = glm::vec4(shadow_cloth_center_, 1.0f);
    pc.params3 = glm::vec4(shadow_cloth_size_, 0.0f, 0.0f);

    // ===== 1) 地面 =====
    if (ground_index_count_ > 0) {
        pc.params0.w = float(TEX_INDEX_GROUND);   // 4

        vkCmdPushConstants(
            cmd,
            pipeline_layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            PUSH_CONSTANT_SIZE,
            &pc
        );

        VkBuffer vb[]       = { ground_vertex_buffer_ };
        VkDeviceSize offs[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
        vkCmdBindIndexBuffer(cmd, ground_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, ground_index_count_, 1, 0, 0, 0);
    }

    // ===== 2) 方块 =====
    if (cube_index_count_ > 0) {
        pc.params0.w = float(TEX_INDEX_CUBE);     // 3

        vkCmdPushConstants(
            cmd,
            pipeline_layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            PUSH_CONSTANT_SIZE,
            &pc
        );

        VkBuffer vb[]       = { cube_vertex_buffer_ };
        VkDeviceSize offs[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
        vkCmdBindIndexBuffer(cmd, cube_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, cube_index_count_, 1, 0, 0, 0);
    }

    // ===== 3) 布 =====
    if (index_count_ > 0) {
        // current_material_ = 0,1,2 （Silk/Heavy/Plastic）
        pc.params0.w = float(current_material_);

        vkCmdPushConstants(
            cmd,
            pipeline_layout_,
            VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            PUSH_CONSTANT_SIZE,
            &pc
        );

        VkBuffer vb[]       = { vertex_buffer_ };
        VkDeviceSize offs[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
        vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(index_count_), 1, 0, 0, 0);
    }

    // ===== 4) ImGui overlay =====
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data && draw_data->CmdListsCount > 0) {
        ImGui_ImplVulkan_RenderDrawData(draw_data, cmd, VK_NULL_HANDLE);
    }

    vkCmdEndRenderPass(cmd);

    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer");
    }
}


bool VulkanRenderer::create_descriptor_set_layout()
{
    VkDescriptorSetLayoutBinding samplerBinding{};
    samplerBinding.binding         = 0;
    samplerBinding.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    samplerBinding.descriptorCount = MAX_TEXTURES;
    samplerBinding.stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT;
    samplerBinding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings    = &samplerBinding;

    if (vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptor_set_layout_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout\n";
        return false;
    }
    return true;
}

void VulkanRenderer::set_cloth_material(int index)
{
    if (index < 0 || index >= CLOTH_TEX_COUNT) return;
    if (current_material_ == index) return;

    current_material_ = index;
}

bool VulkanRenderer::create_cloth_sampler()
{
    VkSamplerCreateInfo info{};
    info.sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter    = VK_FILTER_LINEAR;
    info.minFilter    = VK_FILTER_LINEAR;
    info.mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.mipLodBias   = 0.0f;

    info.anisotropyEnable = VK_FALSE;
    info.maxAnisotropy    = 1.0f;

    info.compareEnable    = VK_FALSE;
    info.compareOp        = VK_COMPARE_OP_ALWAYS;

    info.minLod           = 0.0f;
    info.maxLod           = 0.0f;

    info.borderColor      = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;

    if (vkCreateSampler(device_, &info, nullptr, &cloth_sampler_) != VK_SUCCESS) {
        std::cerr << "Failed to create cloth sampler\n";
        return false;
    }
    return true;
}


bool VulkanRenderer::create_descriptor_pool()
{
    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSize.descriptorCount = MAX_TEXTURES;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = 1;

    if (vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptor_pool_) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }
    return true;
}

bool VulkanRenderer::create_descriptor_set()
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = descriptor_pool_;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &descriptor_set_layout_;

    if (vkAllocateDescriptorSets(device_, &allocInfo, &descriptor_set_) != VK_SUCCESS) {
        std::cerr << "Failed to allocate descriptor set\n";
        return false;
    }

    std::array<VkDescriptorImageInfo, MAX_TEXTURES> imageInfos{};
    for (int i = 0; i < MAX_TEXTURES; ++i) {
        imageInfos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfos[i].imageView   = cloth_image_views_[i];
        imageInfos[i].sampler     = cloth_sampler_;
    }

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = descriptor_set_;
    write.dstBinding      = 0;
    write.dstArrayElement = 0;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = MAX_TEXTURES;
    write.pImageInfo      = imageInfos.data();

    vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

    return true;
}

void VulkanRenderer::draw_frame()
{
    vkWaitForFences(device_, 1, &in_flight_fences_[current_frame_], VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &in_flight_fences_[current_frame_]);

    uint32_t imageIndex;
    VkResult res = vkAcquireNextImageKHR(
        device_, swapchain_, UINT64_MAX,
        image_available_semaphores_[current_frame_],
        VK_NULL_HANDLE, &imageIndex);

    if (res != VK_SUCCESS) {
        return;
    }

    vkResetCommandBuffer(command_buffers_[imageIndex], 0);
    record_command_buffer(imageIndex);

    VkSemaphore waitSemaphores[]   = { image_available_semaphores_[current_frame_] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    VkSemaphore signalSemaphores[] = { render_finished_semaphores_[current_frame_] };

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount   = 1;
    submitInfo.pWaitSemaphores      = waitSemaphores;
    submitInfo.pWaitDstStageMask    = waitStages;
    submitInfo.commandBufferCount   = 1;
    submitInfo.pCommandBuffers      = &command_buffers_[imageIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    VkResult submitRes = vkQueueSubmit(
        graphics_queue_, 1, &submitInfo, in_flight_fences_[current_frame_]);

    if (submitRes != VK_SUCCESS) {
        std::cerr << "Failed to submit draw command buffer, VkResult = "
                  << static_cast<int>(submitRes) << std::endl;
        return;
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;
    presentInfo.swapchainCount     = 1;
    presentInfo.pSwapchains        = &swapchain_;
    presentInfo.pImageIndices      = &imageIndex;

    vkQueuePresentKHR(present_queue_, &presentInfo);

    current_frame_ = (current_frame_ + 1) % MAX_FRAMES_IN_FLIGHT;
}

uint32_t VulkanRenderer::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memProps);

    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((type_filter & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type");
}

void VulkanRenderer::cleanup_vertex_index_buffers()
{
    if (vertex_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vertex_buffer_ = VK_NULL_HANDLE;
    }
    if (vertex_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
        vertex_buffer_memory_ = VK_NULL_HANDLE;
    }
    vertex_buffer_size_ = 0;

    if (index_buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_, index_buffer_, nullptr);
        index_buffer_ = VK_NULL_HANDLE;
    }
    if (index_buffer_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, index_buffer_memory_, nullptr);
        index_buffer_memory_ = VK_NULL_HANDLE;
    }
    index_buffer_size_ = 0;
    index_count_ = 0;
}

void VulkanRenderer::cleanup_swapchain()
{
    for (auto fb : framebuffers_) {
        if (fb != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(device_, fb, nullptr);
        }
    }
    framebuffers_.clear();

    for (auto view : swapchain_image_views_) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device_, view, nullptr);
        }
    }
    swapchain_image_views_.clear();
    swapchain_images_.clear();

    if (depth_image_view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, depth_image_view_, nullptr);
        depth_image_view_ = VK_NULL_HANDLE;
    }
    if (depth_image_ != VK_NULL_HANDLE) {
        vkDestroyImage(device_, depth_image_, nullptr);
        depth_image_ = VK_NULL_HANDLE;
    }
    if (depth_image_memory_ != VK_NULL_HANDLE) {
        vkFreeMemory(device_, depth_image_memory_, nullptr);
        depth_image_memory_ = VK_NULL_HANDLE;
    }

    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

bool VulkanRenderer::create_cloth_textures()
{
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    for (int i = 0; i < MAX_TEXTURES; ++i) {

        // ===== 1) 用 stb_image 读取 PNG =====
        std::string path;
        if (i < CLOTH_TEX_COUNT) {
            // 0,1,2 -> 布的纹理
            path = "../assets/textures/" + std::to_string(i) + ".png";
        } else if (i == TEX_INDEX_CUBE) {
            // 3 -> 方块
            path = "../assets/textures/kaust.png";
        } else if (i == TEX_INDEX_GROUND) {
            // 4 -> 地面
            path = "../assets/textures/ground.png";
        } else {
            path = "../assets/textures/0.png";
        }

        int texWidth = 0;
        int texHeight = 0;
        int texChannels = 0;
        stbi_uc* pixels = stbi_load(path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        if (!pixels) {
            std::cerr << "Failed to load texture: " << path << "\n";
            return false;
        }

        // std::cerr << "Loaded texture " << path
        //           << " (" << texWidth << "x" << texHeight
        //           << ", channels=" << texChannels << ")\n";

        VkDeviceSize imageSize = static_cast<VkDeviceSize>(texWidth) *
                                 static_cast<VkDeviceSize>(texHeight) * 4;
        

        // ===== 2) 创建 staging buffer（HOST_VISIBLE，TRANSFER_SRC） =====
        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;

        {
            VkBufferCreateInfo bufInfo{};
            bufInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufInfo.size  = imageSize;
            bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(device_, &bufInfo, nullptr, &stagingBuffer) != VK_SUCCESS) {
                std::cerr << "Failed to create staging buffer for cloth texture " << i << "\n";
                stbi_image_free(pixels);
                return false;
            }

            VkMemoryRequirements memReq;
            vkGetBufferMemoryRequirements(device_, stagingBuffer, &memReq);

            VkMemoryAllocateInfo alloc{};
            alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc.allocationSize  = memReq.size;
            alloc.memoryTypeIndex = find_memory_type(
                memReq.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

            if (vkAllocateMemory(device_, &alloc, nullptr, &stagingMemory) != VK_SUCCESS) {
                std::cerr << "Failed to allocate staging buffer memory " << i << "\n";
                stbi_image_free(pixels);
                return false;
            }

            vkBindBufferMemory(device_, stagingBuffer, stagingMemory, 0);

            void* data = nullptr;
            vkMapMemory(device_, stagingMemory, 0, imageSize, 0, &data);
            std::memcpy(data, pixels, static_cast<size_t>(imageSize));
            vkUnmapMemory(device_, stagingMemory);
        }

        // 原始像素不用了
        stbi_image_free(pixels);

        // ===== 3) 创建 GPU image（optimal tiling，device local） =====
        {
            VkImageCreateInfo imgInfo{};
            imgInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imgInfo.imageType     = VK_IMAGE_TYPE_2D;
            imgInfo.extent.width  = static_cast<uint32_t>(texWidth);
            imgInfo.extent.height = static_cast<uint32_t>(texHeight);
            imgInfo.extent.depth  = 1;
            imgInfo.mipLevels     = 1;
            imgInfo.arrayLayers   = 1;
            imgInfo.format        = format;
            imgInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;  // ★ optimal
            imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imgInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imgInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
            imgInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateImage(device_, &imgInfo, nullptr, &cloth_images_[i]) != VK_SUCCESS) {
                std::cerr << "Failed to create cloth image " << i << "\n";
                vkDestroyBuffer(device_, stagingBuffer, nullptr);
                vkFreeMemory(device_, stagingMemory, nullptr);
                return false;
            }

            VkMemoryRequirements memReq;
            vkGetImageMemoryRequirements(device_, cloth_images_[i], &memReq);

            VkMemoryAllocateInfo alloc{};
            alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            alloc.allocationSize  = memReq.size;
            alloc.memoryTypeIndex = find_memory_type(
                memReq.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

            if (vkAllocateMemory(device_, &alloc, nullptr, &cloth_image_memory_[i]) != VK_SUCCESS) {
                std::cerr << "Failed to allocate cloth image memory " << i << "\n";
                vkDestroyBuffer(device_, stagingBuffer, nullptr);
                vkFreeMemory(device_, stagingMemory, nullptr);
                return false;
            }

            vkBindImageMemory(device_, cloth_images_[i], cloth_image_memory_[i], 0);
        }

        // ===== 4) 用 single-time command buffer 做 layout 转换 + 拷贝 =====
        VkCommandBuffer cmd = begin_single_time_commands();

        // UNDEFINED -> TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier barrier{};
        barrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout           = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout           = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image               = cloth_images_[i];
        barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel   = 0;
        barrier.subresourceRange.levelCount     = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount     = 1;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        // buffer -> image
        VkBufferImageCopy region{};
        region.bufferOffset                    = 0;
        region.bufferRowLength                 = 0;
        region.bufferImageHeight               = 0;
        region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel       = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount     = 1;
        region.imageOffset                     = {0, 0, 0};
        region.imageExtent                     = {
            static_cast<uint32_t>(texWidth),
            static_cast<uint32_t>(texHeight),
            1
        };

        vkCmdCopyBufferToImage(
            cmd,
            stagingBuffer,
            cloth_images_[i],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        barrier.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout     = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(
            cmd,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );

        end_single_time_commands(cmd);

        // staging buffer 不再需要
        vkDestroyBuffer(device_, stagingBuffer, nullptr);
        vkFreeMemory(device_, stagingMemory, nullptr);

        // ===== 5) 为这个 image 创建 image view =====
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image    = cloth_images_[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format   = format;
        viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel   = 0;
        viewInfo.subresourceRange.levelCount     = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount     = 1;

        if (vkCreateImageView(device_, &viewInfo, nullptr, &cloth_image_views_[i]) != VK_SUCCESS) {
            std::cerr << "Failed to create cloth image view " << i << "\n";
            return false;
        }
    }

    return true;
}

VkCommandBuffer VulkanRenderer::begin_single_time_commands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = command_pool_;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd;
    if (vkAllocateCommandBuffers(device_, &allocInfo, &cmd) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate single-time command buffer");
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(cmd, &beginInfo);
    return cmd;
}

void VulkanRenderer::end_single_time_commands(VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submitInfo{};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &cmd;

    vkQueueSubmit(graphics_queue_, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphics_queue_);

    vkFreeCommandBuffers(device_, command_pool_, 1, &cmd);
}


void VulkanRenderer::set_shadow_params(const glm::vec3& lightDir,
                                       const glm::vec3& boxCenter,
                                       float boxRadius,
                                       const glm::vec3& clothCenter,
                                       const glm::vec2& clothSize)
{
    light_dir_          = glm::normalize(lightDir);
    shadow_box_center_  = boxCenter;
    shadow_box_radius_  = boxRadius;
    shadow_cloth_center_ = clothCenter;
    shadow_cloth_size_   = clothSize;
}

