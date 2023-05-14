#pragma once
#include "device.h"

namespace grassland::vulkan {
class Fence {
 public:
  GRASSLAND_CANNOT_COPY(Fence)
  Fence(const class Device &device, bool signaled);
  ~Fence();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkFence, fence_)
};
}  // namespace grassland::vulkan
