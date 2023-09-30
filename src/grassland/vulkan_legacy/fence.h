#pragma once
#include <grassland/vulkan_legacy/device.h>

namespace grassland::vulkan_legacy {
class Fence {
 public:
  explicit Fence(Device *device);
  ~Fence();

 private:
  GRASSLAND_VULKAN_HANDLE(VkFence)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
