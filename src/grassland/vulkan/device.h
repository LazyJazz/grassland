#pragma once
#include <grassland/vulkan/physical_device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Device {
 public:
  Device(PhysicalDevice *physical_device, Surface *surface);
  ~Device();

 private:
  VK_HANDLE(VkDevice)
};
}  // namespace grassland::vulkan
