#pragma once
#include "device.h"

namespace grassland::vulkan {
class Semaphore {
 public:
  GRASSLAND_CANNOT_COPY(Semaphore)
  Semaphore(const class Device &device);
  ~Semaphore();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkSemaphore, semaphore_)
};
}  // namespace grassland::vulkan
