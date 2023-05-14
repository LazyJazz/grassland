#include "fence.h"

namespace grassland::vulkan {
Fence::Fence(const class Device &device, bool signaled) : device_(device) {
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_info.flags = signaled ? VK_FENCE_CREATE_SIGNALED_BIT : 0;

  GRASSLAND_VULKAN_CHECK(
      vkCreateFence(device_.Handle(), &fence_info, nullptr, &fence_));
}

Fence::~Fence() {
  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_.Handle(), fence_, nullptr);
  }
}
}  // namespace grassland::vulkan
