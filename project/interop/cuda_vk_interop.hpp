#pragma once
#include <vulkan/vulkan.h>

struct CudaVkSharedBuffer {
  VkBuffer vk_buffer{VK_NULL_HANDLE};
  void*    cuda_ptr{nullptr};
  size_t   bytes{0};
};

struct CudaVkInterop {
  bool register_buffer(VkDevice dev, VkBuffer buf, size_t bytes, CudaVkSharedBuffer& out);
  void unregister_buffer(CudaVkSharedBuffer& buf);
  void sync_to_cuda(VkQueue queue);
  void sync_to_vulkan(VkQueue queue);
};
