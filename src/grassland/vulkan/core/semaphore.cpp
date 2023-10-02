#include "grassland/vulkan/core/semaphore.h"

namespace grassland::vulkan {
// Complete semaphore class
Semaphore::Semaphore(Device *device) : device_(device) {
  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = nullptr;
  semaphore_create_info.flags = 0;
  if (vkCreateSemaphore(device_->Handle(), &semaphore_create_info, nullptr,
                        &semaphore_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create semaphore!");
  }
}
Semaphore::~Semaphore() {
  vkDestroySemaphore(device_->Handle(), semaphore_, nullptr);
}

VkSemaphore Semaphore::Handle() const {
  return semaphore_;
}
}  // namespace grassland::vulkan
