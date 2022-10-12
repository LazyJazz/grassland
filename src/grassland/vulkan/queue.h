#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Queue {
 public:
  Queue(Device *device, uint32_t queue_family_index);
  ~Queue();

 private:
  GRASSLAND_VULKAN_HANDLE(VkQueue)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
