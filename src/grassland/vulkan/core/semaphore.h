#pragma once
#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {
class Semaphore {
 public:
  explicit Semaphore(Device *device);
  ~Semaphore();

  [[nodiscard]] VkSemaphore Handle() const;

 private:
  Device *device_{};
  VkSemaphore semaphore_{};
};
}  // namespace grassland::vulkan
