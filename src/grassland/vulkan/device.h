#pragma once
#include <grassland/vulkan/physical_device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Device {
 public:
  Device(PhysicalDevice *physical_device, Surface *surface);
  ~Device();
  PhysicalDevice *GetPhysicalDevice();
  Surface *GetSurface();

 private:
  GRASSLAND_VULKAN_HANDLE(VkDevice)
  PhysicalDevice *physical_device_{nullptr};
  Surface *surface_{nullptr};
};
}  // namespace grassland::vulkan
