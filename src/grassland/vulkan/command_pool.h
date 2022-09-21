#pragma once

#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class CommandPool {
 public:
  explicit CommandPool(Device *device);
  ~CommandPool();
  Device *GetDevice();

 private:
  VK_HANDLE(VkCommandPool)
  Device *device_{nullptr};
};
}  // namespace grassland::vulkan
