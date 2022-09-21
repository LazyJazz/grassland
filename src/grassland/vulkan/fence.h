#pragma once
#include <grassland/vulkan/device.h>

namespace grassland::vulkan {
class Fence {
 public:
  Fence();

 private:
  GRASSLAND_VULKAN_HANDLE(VkFence)
};
}  // namespace grassland::vulkan
