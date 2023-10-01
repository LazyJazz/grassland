#pragma once
#include "grassland/vulkan/core/instance.h"

namespace grassland::vulkan {
class Device;
class Queue {
 public:
  explicit Queue(Device *device = nullptr, uint32_t queue_family_index = 0);

  [[nodiscard]] VkQueue Handle() const;
  class Device *Device() {
    return device_;
  }

  void WaitIdle() const;

 private:
  class Device *device_{};
  VkQueue queue_{};
};
}  // namespace grassland::vulkan
