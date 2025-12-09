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

// =================== 辅助结构 ===================
struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 color;
};

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
    if (!create_render_pass())     return false;
    if (!create_graphics_pipeline()) return false;
    if (!create_framebuffers())    return false;
    if (!create_command_pool())    return false;
    if (!allocate_command_buffers()) return false;
    if (!create_sync_objects())    return false;

    // 先创建一个空的顶点/索引缓冲（后面 update_mesh 会重建）
    create_vertex_buffer(sizeof(Vertex) * 1);
    create_index_buffer(sizeof(uint32_t) * 1);

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

    // record_command_buffers();
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
    // 这里暂时使用 window 的大小
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
    raster.depthBiasEnable         = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable  = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.logicOpEnable   = VK_FALSE;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments    = &colorBlendAttachment;

    // VkPipelineLayoutCreateInfo layoutInfo{};
    // layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // layoutInfo.setLayoutCount = 0;
    // layoutInfo.pushConstantRangeCount = 0;

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 0;
    layoutInfo.pSetLayouts            = nullptr;
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
    depth_format_ = VK_FORMAT_D32_SFLOAT; // 简化版，直接用这个

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
    // 1. 构建 8 个顶点（世界空间位置）
    std::vector<Vertex> verts(8);
    const float cx = center.x, cy = center.y, cz = center.z;
    const float hx = half_extent.x, hy = half_extent.y, hz = half_extent.z;

    // front (z -)
    verts[0].pos = Vec3(cx - hx, cy - hy, cz - hz);
    verts[1].pos = Vec3(cx + hx, cy - hy, cz - hz);
    verts[2].pos = Vec3(cx + hx, cy + hy, cz - hz);
    verts[3].pos = Vec3(cx - hx, cy + hy, cz - hz);
    // back (z +)
    verts[4].pos = Vec3(cx - hx, cy - hy, cz + hz);
    verts[5].pos = Vec3(cx + hx, cy - hy, cz + hz);
    verts[6].pos = Vec3(cx + hx, cy + hy, cz + hz);
    verts[7].pos = Vec3(cx - hx, cy + hy, cz + hz);

    // 法线先清零，下面按三角面累加
    for (auto& v : verts) {
        v.normal = Vec3(0.0f);
    }

    // 2. 12 个三角（36 indices）
    std::vector<u32> indices = {
        // front (-Z)
        0, 1, 2,  2, 3, 0,
        // back (+Z)
        5, 4, 7,  7, 6, 5,
        // left (-X)
        4, 0, 3,  3, 7, 4,
        // right (+X)
        1, 5, 6,  6, 2, 1,
        // bottom (-Y)
        4, 5, 1,  1, 0, 4,
        // top (+Y)
        3, 2, 6,  6, 7, 3
    };

    // 3. 用三角面累加 vertex normal
    for (size_t i = 0; i < indices.size(); i += 3) {
        u32 i0 = indices[i + 0];
        u32 i1 = indices[i + 1];
        u32 i2 = indices[i + 2];

        Vec3 p0 = verts[i0].pos;
        Vec3 p1 = verts[i1].pos;
        Vec3 p2 = verts[i2].pos;

        Vec3 e1 = p1 - p0;
        Vec3 e2 = p2 - p0;
        Vec3 n  = glm::cross(e1, e2);

        verts[i0].normal += n;
        verts[i1].normal += n;
        verts[i2].normal += n;
    }
    for (auto& v : verts) {
        float len = glm::length(v.normal);
        if (len > 1e-6f) v.normal /= len;
        else             v.normal = Vec3(0, 1, 0);
    }

    // 4. 确保有足够大的 cube vertex/index buffer（host-visible）
    VkDeviceSize vbytes = sizeof(Vertex) * verts.size();
    VkDeviceSize ibytes = sizeof(u32)    * indices.size();

    // --- 顶点缓冲 ---
    if (cube_vertex_buffer_ == VK_NULL_HANDLE || vbytes > cube_vertex_buffer_size_) {
        if (cube_vertex_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, cube_vertex_buffer_, nullptr);
            vkFreeMemory(device_, cube_vertex_memory_, nullptr);
            cube_vertex_buffer_ = VK_NULL_HANDLE;
            cube_vertex_memory_ = VK_NULL_HANDLE;
            cube_vertex_buffer_size_ = 0;
        }

        cube_vertex_buffer_size_ = vbytes;

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = vbytes;
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &cube_vertex_buffer_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create cube vertex buffer");
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, cube_vertex_buffer_, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &cube_vertex_memory_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate cube vertex buffer memory");
        }

        vkBindBufferMemory(device_, cube_vertex_buffer_, cube_vertex_memory_, 0);
    }

    // --- 索引缓冲 ---
    if (cube_index_buffer_ == VK_NULL_HANDLE || ibytes > cube_index_count_ * sizeof(u32)) {
        if (cube_index_buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, cube_index_buffer_, nullptr);
            vkFreeMemory(device_, cube_index_memory_, nullptr);
            cube_index_buffer_ = VK_NULL_HANDLE;
            cube_index_memory_ = VK_NULL_HANDLE;
            cube_index_count_  = 0;
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size  = ibytes;
        bufferInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device_, &bufferInfo, nullptr, &cube_index_buffer_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create cube index buffer");
        }

        VkMemoryRequirements memReq;
        vkGetBufferMemoryRequirements(device_, cube_index_buffer_, &memReq);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize  = memReq.size;
        allocInfo.memoryTypeIndex = find_memory_type(
            memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if (vkAllocateMemory(device_, &allocInfo, nullptr, &cube_index_memory_) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate cube index buffer memory");
        }

        vkBindBufferMemory(device_, cube_index_buffer_, cube_index_memory_, 0);
    }

    // 5. 拷贝数据 host → device（因为是 HOST_VISIBLE，就直接 memcpy）
    void* data = nullptr;
    // vertex
    vkMapMemory(device_, cube_vertex_memory_, 0, vbytes, 0, &data);
    std::memcpy(data, verts.data(), vbytes);
    vkUnmapMemory(device_, cube_vertex_memory_);

    // index
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

    // 地面法线向上
    for (auto& v : verts) {
        v.normal = Vec3(0.0f, 1.0f, 0.0f);
    }

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

void VulkanRenderer::record_command_buffers()
{
    for (size_t i = 0; i < command_buffers_.size(); ++i) {
        VkCommandBuffer cmd = command_buffers_[i];

        // 开始录制 command buffer
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0;
        beginInfo.pInheritanceInfo = nullptr;

        if (vkBeginCommandBuffer(cmd, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer");
        }

        // 设置 render pass begin
        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass  = render_pass_;
        rpBegin.framebuffer = framebuffers_[i];
        rpBegin.renderArea.offset = {0, 0};
        rpBegin.renderArea.extent = swapchain_extent_;

        // 清颜色 + 深度
        VkClearValue clearValues[2];
        clearValues[0].color        = {{0.02f, 0.02f, 0.05f, 1.0f}};  // 背景色
        clearValues[1].depthStencil = {1.0f, 0};                       // 深度=1.0

        rpBegin.clearValueCount = 2;
        rpBegin.pClearValues    = clearValues;

        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        // 绑定图形管线
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

        // Push 常量结构：MVP + 颜色
        PushConstants pc{};
        pc.mvp = mvp_;

        // ===== 1) 画地面 =====
        if (ground_index_count_ > 0) {
            pc.color = glm::vec4(0.20f, 0.25f, 0.20f, 1.0f);  // 偏绿色地板

            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(PushConstants),
                &pc
            );

            VkBuffer vb[]       = { ground_vertex_buffer_ };
            VkDeviceSize offs[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
            vkCmdBindIndexBuffer(cmd, ground_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, ground_index_count_, 1, 0, 0, 0);
        }

        // ===== 2) 画方块 =====
        if (cube_index_count_ > 0) {
            pc.color = glm::vec4(0.85f, 0.35f, 0.10f, 1.0f);  // 比较亮的橙红色方块

            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(PushConstants),
                &pc
            );

            VkBuffer vb[]       = { cube_vertex_buffer_ };
            VkDeviceSize offs[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
            vkCmdBindIndexBuffer(cmd, cube_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(cmd, cube_index_count_, 1, 0, 0, 0);
        }

        // ===== 3) 画布 =====
        if (index_count_ > 0) {
            pc.color = glm::vec4(0.40f, 0.40f, 0.90f, 1.0f);  // 布：偏蓝一点

            vkCmdPushConstants(
                cmd,
                pipeline_layout_,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(PushConstants),
                &pc
            );

            VkBuffer vb[]       = { vertex_buffer_ };
            VkDeviceSize offs[] = { 0 };
            vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
            vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);

            vkCmdDrawIndexed(
                cmd,
                static_cast<uint32_t>(index_count_),
                1,
                0, 0, 0
            );
        }

        // 结束 render pass
        vkCmdEndRenderPass(cmd);

        // 结束 command buffer 录制
        if (vkEndCommandBuffer(cmd) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer");
        }
    }
}

        // // Push MVP 给 vertex shader 的 push_constant
        // vkCmdPushConstants(
        //     cmd,
        //     pipeline_layout_,
        //     VK_SHADER_STAGE_VERTEX_BIT,
        //     0,
        //     sizeof(glm::mat4),
        //     &mvp_
        // );

        // // 1) 地面
        // if (ground_index_count_ > 0) {
        //     VkBuffer vb[] = { ground_vertex_buffer_ };
        //     VkDeviceSize offs[] = { 0 };
        //     vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
        //     vkCmdBindIndexBuffer(cmd, ground_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        //     vkCmdDrawIndexed(cmd, ground_index_count_, 1, 0, 0, 0);
        // }

        // // 2) 方块
        // if (cube_index_count_ > 0) {
        //     VkBuffer vb[] = { cube_vertex_buffer_ };
        //     VkDeviceSize offs[] = { 0 };
        //     vkCmdBindVertexBuffers(cmd, 0, 1, vb, offs);
        //     vkCmdBindIndexBuffer(cmd, cube_index_buffer_, 0, VK_INDEX_TYPE_UINT32);
        //     vkCmdDrawIndexed(cmd, cube_index_count_, 1, 0, 0, 0);
        // }

        // VkBuffer vertexBuffers[] = { vertex_buffer_ };
        // VkDeviceSize offsets[]   = { 0 };
        // vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

        // vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);

        // if (index_count_ > 0) {
        //     vkCmdDrawIndexed(cmd, static_cast<uint32_t>(index_count_), 1, 0, 0, 0);
        // }

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

    if (!cb_recorded_ && index_count_ > 0) {
        record_command_buffers();
        cb_recorded_ = true;
    }

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
