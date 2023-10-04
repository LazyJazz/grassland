#include "grassland/vulkan/core/fence.h"

namespace grassland::vulkan {
Fence::Fence(Device *device) : device_(device) {
  VkFenceCreateInfo fence_create_info{};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.pNext = nullptr;
  fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  if (vkCreateFence(device_->Handle(), &fence_create_info, nullptr, &fence_) !=
      VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create fence!");
  }
}

Fence::~Fence() {
  vkDestroyFence(device_->Handle(), fence_, nullptr);
}

VkFence Fence::Handle() const {
  return fence_;
}
}  // namespace grassland::vulkan
