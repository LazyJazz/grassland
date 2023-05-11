#pragma once

#include "device.h"

namespace grassland::vulkan {
class CommandPool {
 public:
  GRASSLAND_CANNOT_COPY(CommandPool)
  explicit CommandPool(const class Device &device);
  CommandPool(const class Device &device, int queue_family_index);
  ~CommandPool();

 private:
  GRASSLAND_VULKAN_HANDLE(VkCommandPool, command_pool_)
  GRASSLAND_VULKAN_DEVICE
};
}  // namespace grassland::vulkan
