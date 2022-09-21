#pragma once

#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class CommandPool {
 public:
  explicit CommandPool(Device *device);
  ~CommandPool();

 private:
  VK_HANDLE(VkCommandPool)
  VK_DEVICE_PTR
};
}  // namespace grassland::vulkan
