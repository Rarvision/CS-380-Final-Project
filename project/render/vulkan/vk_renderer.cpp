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

// =================== 辅助结构 ===================

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
    wait_idle();

    cleanup_swapchain();

    if (vertex_buffer_) {
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
        vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
    }
    if (index_buffer_) {
        vkDestroyBuffer(device_, index_buffer_, nullptr);
        vkFreeMemory(device_, index_buffer_memory_, nullptr);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (image_available_semaphores_[i]) {
            vkDestroySemaphore(device_, image_available_semaphores_[i], nullptr);
        }
        if (render_finished_semaphores_[i]) {
            vkDestroySemaphore(device_, render_finished_semaphores_[i], nullptr);
        }
        if (in_flight_fences_[i]) {
            vkDestroyFence(device_, in_flight_fences_[i], nullptr);
        }
    }

    if (command_pool_) {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }

    if (device_) {
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
    }
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
    if (!create_render_pass())     return false;
    if (!create_graphics_pipeline()) return false;
    if (!create_framebuffers())    return false;
    if (!create_command_pool())    return false;
    if (!allocate_command_buffers()) return false;
    if (!create_sync_objects())    return false;

    // 先创建一个空的顶点/索引缓冲（后面 update_mesh 会重建）
    create_vertex_buffer(sizeof(Vertex) * 1);
    create_index_buffer(sizeof(uint32_t) * 1);

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

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments    = &colorRef;

    VkRenderPassCreateInfo rp{};
    rp.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rp.attachmentCount = 1;
    rp.pAttachments    = &color;
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
    // 读取 SPIR-V 着色器（你需要先用 glslc 编译）
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

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 0;
    layoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(device_, &layoutInfo, nullptr, &pipeline_layout_) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout\n";
        return false;
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = 2;
    pipelineInfo.pStages             = stages;
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState   = &multisample;
    pipelineInfo.pDepthStencilState  = nullptr;
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
        VkImageView attachments[] = { swapchain_image_views_[i] };

        VkFramebufferCreateInfo info{};
        info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass      = render_pass_;
        info.attachmentCount = 1;
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
        std::cerr << "Failed to create vertex buffer\n";
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
        std::cerr << "Failed to allocate vertex buffer memory\n";
        return false;
    }

    vkBindBufferMemory(device_, vertex_buffer_, vertex_buffer_memory_, 0);
    return true;
}

bool VulkanRenderer::create_index_buffer(size_t size_bytes)
{
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size  = size_bytes;
    info.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device_, &info, nullptr, &index_buffer_) != VK_SUCCESS) {
        std::cerr << "Failed to create index buffer\n";
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
        std::cerr << "Failed to allocate index buffer memory\n";
        return false;
    }

    vkBindBufferMemory(device_, index_buffer_, index_buffer_memory_, 0);
    return true;
}

void VulkanRenderer::update_mesh(const std::vector<Vertex>& vertices,
                                 const std::vector<u32>& indices)
{
//    std::cerr << "[VK] update_mesh: vertices=" << vertices.size()
//              << " indices=" << indices.size() << std::endl;

    size_t vbytes = vertices.size() * sizeof(Vertex);
    size_t ibytes = indices.size()  * sizeof(u32);

    // 如果 buffer 不够大，就重建
    if (!vertex_buffer_ || vbytes > vertex_buffer_size_) {
        if (vertex_buffer_) {
            vkDestroyBuffer(device_, vertex_buffer_, nullptr);
            vkFreeMemory(device_, vertex_buffer_memory_, nullptr);
        }
        create_vertex_buffer(vbytes);
    }
    if (!index_buffer_ || ibytes > index_count_ * sizeof(u32)) {
        if (index_buffer_) {
            vkDestroyBuffer(device_, index_buffer_, nullptr);
            vkFreeMemory(device_, index_buffer_memory_, nullptr);
        }
        create_index_buffer(ibytes);
    }

    // CPU → GPU（host-visible）
    void* data = nullptr;
    vkMapMemory(device_, vertex_buffer_memory_, 0, vbytes, 0, &data);
    std::memcpy(data, vertices.data(), vbytes);
    vkUnmapMemory(device_, vertex_buffer_memory_);

    vkMapMemory(device_, index_buffer_memory_, 0, ibytes, 0, &data);
    std::memcpy(data, indices.data(), ibytes);
    vkUnmapMemory(device_, index_buffer_memory_);

    index_count_ = indices.size();

    // 更新完顶点后，重新录制 command buffer（简单处理）
    // record_command_buffers();
}

// =================== 命令缓冲 & draw ===================

void VulkanRenderer::record_command_buffers()
{
    for (size_t i = 0; i < command_buffers_.size(); ++i) {
        VkCommandBuffer cmd = command_buffers_[i];

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        vkBeginCommandBuffer(cmd, &beginInfo);

        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType       = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass  = render_pass_;
        rpBegin.framebuffer = framebuffers_[i];
        rpBegin.renderArea.offset = {0, 0};
        rpBegin.renderArea.extent = swapchain_extent_;

        VkClearValue clearColor{};
        clearColor.color = {{0.02f, 0.02f, 0.05f, 1.0f}};
        rpBegin.clearValueCount = 1;
        rpBegin.pClearValues    = &clearColor;

        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);

        VkBuffer vertexBuffers[] = { vertex_buffer_ };
        VkDeviceSize offsets[]   = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);

        vkCmdBindIndexBuffer(cmd, index_buffer_, 0, VK_INDEX_TYPE_UINT32);

        if (index_count_ > 0) {
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(index_count_), 1, 0, 0, 0);
        }

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }
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

void VulkanRenderer::cleanup_swapchain()
{
    for (auto fb : framebuffers_) {
        vkDestroyFramebuffer(device_, fb, nullptr);
    }
    framebuffers_.clear();

    if (graphics_pipeline_) {
        vkDestroyPipeline(device_, graphics_pipeline_, nullptr);
        graphics_pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_) {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (render_pass_) {
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        render_pass_ = VK_NULL_HANDLE;
    }

    for (auto view : swapchain_image_views_) {
        vkDestroyImageView(device_, view, nullptr);
    }
    swapchain_image_views_.clear();

    if (swapchain_) {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}
