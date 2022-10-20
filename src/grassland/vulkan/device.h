#pragma once
#include <grassland/vulkan/physical_device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {

class Queue;

class Device {
 public:
  explicit Device(PhysicalDevice *physical_device);
  Device(PhysicalDevice *physical_device, Surface *surface);
  ~Device();
  PhysicalDevice *GetPhysicalDevice();
  Surface *GetSurface();
  Queue *GetGraphicsQueue();
  void WaitIdle();

 private:
  GRASSLAND_VULKAN_HANDLE(VkDevice)
  PhysicalDevice *physical_device_{nullptr};
  Surface *surface_{nullptr};
  Queue *graphics_queue_{nullptr};
};
}  // namespace grassland::vulkan
