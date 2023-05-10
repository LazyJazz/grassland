#pragma once
#include "surface.h"

namespace grassland::vulkan {

class PhysicalDevice {
 public:
  PhysicalDevice(VkPhysicalDevice physical_device = VK_NULL_HANDLE);
  [[nodiscard]] std::string DeviceName() const;
  [[nodiscard]] VkPhysicalDeviceProperties GetProperties() const;
  [[nodiscard]] VkPhysicalDeviceFeatures GetFeatures() const;
  [[nodiscard]] VkPhysicalDeviceRayTracingPipelinePropertiesKHR
  GetRayTracingProperties() const;
  [[nodiscard]] VkPhysicalDeviceRayTracingPipelineFeaturesKHR
  GetRayTracingFeatures() const;

 private:
  GRASSLAND_VULKAN_HANDLE(VkPhysicalDevice, physical_device_)
};
}  // namespace grassland::vulkan
