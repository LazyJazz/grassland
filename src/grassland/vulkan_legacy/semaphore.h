#pragma once
#include <grassland/vulkan_legacy/device.h>

namespace grassland::vulkan_legacy {
class Semaphore {
 public:
  explicit Semaphore(Device *device);
  ~Semaphore();

 private:
  GRASSLAND_VULKAN_HANDLE(VkSemaphore)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
