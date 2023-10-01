#include "grassland/vulkan/core/queue.h"

#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {
Queue::Queue(grassland::vulkan::Device *device, uint32_t queue_family_index)
    : device_(device) {
  if (device) {
    vkGetDeviceQueue(device->Handle(), queue_family_index, 0, &queue_);
  }
}

VkQueue Queue::Handle() const {
  return queue_;
}

void Queue::WaitIdle() const {
  vkQueueWaitIdle(queue_);
}
}  // namespace grassland::vulkan
