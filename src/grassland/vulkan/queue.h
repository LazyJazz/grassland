#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Queue {
 public:
  Queue(Device *device, uint32_t queue_family_index);
  ~Queue();

 private:
  VK_HANDLE(VkQueue)
};
}  // namespace grassland::vulkan
