#pragma once
#include "grassland/vulkan/core/instance.h"

namespace grassland::vulkan {
class Device;
class Queue {
 public:
  explicit Queue(Device *device, uint32_t queue_family_index);

  [[nodiscard]] VkQueue Handle() const;
  class Device *Device() {
    return device_;
  }

 private:
  class Device *device_{};
  VkQueue queue_{};
};
}  // namespace grassland::vulkan
