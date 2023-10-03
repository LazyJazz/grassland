#include "grassland/vulkan/core/queue.h"

#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {
Queue::Queue(struct Device *device,
             uint32_t queue_family_index,
             uint32_t queue_index)
    : device_(device), queue_family_index_(queue_family_index) {
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
