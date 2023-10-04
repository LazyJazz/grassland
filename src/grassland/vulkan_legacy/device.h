#pragma once
#include <grassland/vulkan_legacy/physical_device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {

class Queue;

class Device {
 public:
  explicit Device(
      PhysicalDevice *physical_device,
      const std::vector<const char *> &extra_device_extensions = {},
      bool enable_validation_layers = vulkan::kDefaultEnableValidationLayers);
  Device(PhysicalDevice *physical_device,
         Surface *surface,
         const std::vector<const char *> &extra_device_extensions = {},
         bool enable_validation_layers = vulkan::kDefaultEnableValidationLayers,
         void *extraDeviceFeatures = nullptr);
  ~Device();
  [[nodiscard]] PhysicalDevice *GetPhysicalDevice() const;
  [[nodiscard]] Surface *GetSurface() const;
  [[nodiscard]] Queue *GetGraphicsQueue() const;
  [[nodiscard]] Queue *GetPresentQueue() const;
  void WaitIdle();

 private:
  GRASSLAND_VULKAN_HANDLE(VkDevice)
  PhysicalDevice *physical_device_{nullptr};
  Surface *surface_{nullptr};
  std::unique_ptr<Queue> graphics_queue_;
  std::unique_ptr<Queue> present_queue_;
};
}  // namespace grassland::vulkan_legacy
