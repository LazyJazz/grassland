#pragma once
#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {
class CommandPool {
 public:
  explicit CommandPool(class Device *device);
  ~CommandPool();

  [[nodiscard]] VkCommandPool Handle() const;
  [[nodiscard]] class Device *Device() const;

 private:
  class Device *device_;
  VkCommandPool command_pool_{};
};
}  // namespace grassland::vulkan
