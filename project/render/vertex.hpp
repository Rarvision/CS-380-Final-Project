#pragma once
#include <array>
#include <vulkan/vulkan.h>
#include "../core/common/types.hpp"

struct Vertex {
    Vec3 pos;
    Vec3 normal;

    static VkVertexInputBindingDescription binding_description() {
        VkVertexInputBindingDescription binding{};
        binding.binding   = 0;
        binding.stride    = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        return binding;
    }

    static std::array<VkVertexInputAttributeDescription, 2> attribute_descriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attrs{};

        // location 0: position
        attrs[0].binding  = 0;
        attrs[0].location = 0;
        attrs[0].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset   = offsetof(Vertex, pos);

        // location 1: normal
        attrs[1].binding  = 0;
        attrs[1].location = 1;
        attrs[1].format   = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset   = offsetof(Vertex, normal);

        return attrs;
    }
};
