#include <grassland/vulkan/queue.h>
#include <vulkan/vulkan.h>

namespace grassland::vulkan {

Queue::Queue(Device *device, uint32_t queue_family_index) {
  vkGetDeviceQueue(device->GetHandle(), queue_family_index, 0, &handle_);
}

Queue::~Queue() = default;

}  // namespace grassland::vulkan
