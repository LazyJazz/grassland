#include "semaphore.h"

namespace grassland::vulkan {

Semaphore::Semaphore(const class Device &device) : device_(device) {
  VkSemaphoreCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  GRASSLAND_VULKAN_CHECK(
      vkCreateSemaphore(device_.Handle(), &create_info, nullptr, &semaphore_));
}

Semaphore::~Semaphore() {
  if (semaphore_ != VK_NULL_HANDLE) {
    vkDestroySemaphore(device_.Handle(), semaphore_, nullptr);
  }
}

}  // namespace grassland::vulkan
