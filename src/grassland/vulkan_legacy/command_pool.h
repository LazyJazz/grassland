#pragma once

#include <grassland/vulkan_legacy/device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class CommandPool {
 public:
  explicit CommandPool(Device *device);
  ~CommandPool();

 private:
  GRASSLAND_VULKAN_HANDLE(VkCommandPool)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
