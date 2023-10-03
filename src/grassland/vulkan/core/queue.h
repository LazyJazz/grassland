#pragma once
#include "grassland/vulkan/core/instance.h"

namespace grassland::vulkan {
class Queue {
 public:
  explicit Queue(struct Device *device = nullptr,
                 uint32_t queue_family_index = 0,
                 uint32_t queue_index = 0);

  [[nodiscard]] VkQueue Handle() const;
  [[nodiscard]] class Device *Device() const {
    return device_;
  }
  [[nodiscard]] uint32_t QueueFamilyIndex() const {
    return queue_family_index_;
  }

  void WaitIdle() const;

 private:
  class Device *device_{};
  uint32_t queue_family_index_{};
  VkQueue queue_{};
};
}  // namespace grassland::vulkan
