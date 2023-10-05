#pragma once

#include "grassland/vulkan/core/instance.h"

namespace grassland::vulkan {

class Surface;

// Physical Device
class PhysicalDevice {
 public:
  explicit PhysicalDevice(VkPhysicalDevice physical_device);
  ~PhysicalDevice();

  [[nodiscard]] VkPhysicalDevice Handle() const;

  // Functions that Return Key Features and Properties
  [[nodiscard]] VkPhysicalDeviceFeatures GetPhysicalDeviceFeatures() const;
  [[nodiscard]] VkPhysicalDeviceProperties GetPhysicalDeviceProperties() const;
  [[nodiscard]] VkPhysicalDeviceMemoryProperties
  GetPhysicalDeviceMemoryProperties() const;
  [[nodiscard]] VkPhysicalDeviceRayTracingPipelinePropertiesKHR
  GetPhysicalDeviceRayTracingPipelineProperties() const;
  [[nodiscard]] VkPhysicalDeviceRayTracingPipelineFeaturesKHR
  GetPhysicalDeviceRayTracingPipelineFeatures() const;

  [[nodiscard]] uint64_t GetDeviceLocalMemorySize() const;
  [[nodiscard]] std::vector<VkExtensionProperties> GetDeviceExtensions() const;

  [[nodiscard]] std::vector<VkQueueFamilyProperties> GetQueueFamilyProperties()
      const;

  bool IsExtensionSupported(const char *extension_name) const;
  [[nodiscard]] bool SupportGeometryShader() const;
  [[nodiscard]] bool SupportRayTracing() const;

  [[nodiscard]] uint64_t Evaluate() const;

  [[nodiscard]] uint32_t GraphicsFamilyIndex() const;
  uint32_t PresentFamilyIndex(Surface *surface) const;

  [[nodiscard]] VkSampleCountFlagBits GetMaxUsableSampleCount() const;

 private:
  VkPhysicalDevice physical_device_{};
};

}  // namespace grassland::vulkan
