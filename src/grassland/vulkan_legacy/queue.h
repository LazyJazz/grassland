#pragma once
#include <grassland/vulkan_legacy/device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class Queue {
 public:
  Queue(Device *device, uint32_t queue_family_index);
  ~Queue();
  void WaitIdle();

 private:
  GRASSLAND_VULKAN_HANDLE(VkQueue)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
