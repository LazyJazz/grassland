#include "grassland/vulkan/core/physical_device.h"

namespace grassland::vulkan {
PhysicalDevice::PhysicalDevice(VkPhysicalDevice physical_device) {
  physical_device_ = physical_device;
}

PhysicalDevice::~PhysicalDevice() = default;

VkPhysicalDevice PhysicalDevice::Handle() const {
  return physical_device_;
}

VkPhysicalDeviceFeatures PhysicalDevice::GetPhysicalDeviceFeatures() const {
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical_device_, &features);
  return features;
}

VkPhysicalDeviceProperties PhysicalDevice::GetPhysicalDeviceProperties() const {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device_, &properties);
  return properties;
}

VkPhysicalDeviceMemoryProperties
PhysicalDevice::GetPhysicalDeviceMemoryProperties() const {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);
  return properties;
}

uint64_t PhysicalDevice::GetDeviceLocalMemorySize() const {
  VkPhysicalDeviceMemoryProperties properties =
      GetPhysicalDeviceMemoryProperties();
  uint64_t device_local_memory_size = 0;
  for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
    if (properties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_local_memory_size = properties.memoryHeaps[i].size;
    }
  }
  return device_local_memory_size;
}

std::vector<VkExtensionProperties> PhysicalDevice::GetDeviceExtensions() const {
  uint32_t extension_count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr,
                                       &extension_count, nullptr);
  std::vector<VkExtensionProperties> extensions(extension_count);
  vkEnumerateDeviceExtensionProperties(physical_device_, nullptr,
                                       &extension_count, extensions.data());
  return extensions;
}

bool PhysicalDevice::IsExtensionSupported(const char *extension_name) const {
  std::vector<VkExtensionProperties> extensions = GetDeviceExtensions();
  for (const auto &extension : extensions) {
    if (strcmp(extension.extensionName, extension_name) == 0) {
      return true;
    }
  }
  return false;
}

[[maybe_unused]] bool PhysicalDevice::IsGeometryShaderSupported() const {
  // Geometry shader is feature of Vulkan
  VkPhysicalDeviceFeatures features = GetPhysicalDeviceFeatures();
  return features.geometryShader;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR
PhysicalDevice::GetPhysicalDeviceRayTracingPipelineProperties() const {
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR properties{};
  properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  properties.pNext = nullptr;
  VkPhysicalDeviceProperties2 properties2{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &properties;
  vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
  return properties;
}

VkPhysicalDeviceRayTracingPipelineFeaturesKHR
PhysicalDevice::GetPhysicalDeviceRayTracingPipelineFeatures() const {
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR features{};
  features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  features.pNext = nullptr;
  VkPhysicalDeviceFeatures2 features2{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &features;
  vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
  return features;
}

bool PhysicalDevice::IsRayTracingSupported() const {
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR features =
      GetPhysicalDeviceRayTracingPipelineFeatures();
  return features.rayTracingPipeline;
}

uint64_t PhysicalDevice::EvaluateDeviceScore() const {
  uint64_t score = 0;
  VkPhysicalDeviceProperties properties = GetPhysicalDeviceProperties();
  VkPhysicalDeviceFeatures features = GetPhysicalDeviceFeatures();
  VkPhysicalDeviceMemoryProperties memory_properties =
      GetPhysicalDeviceMemoryProperties();
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties =
      GetPhysicalDeviceRayTracingPipelineProperties();
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_features =
      GetPhysicalDeviceRayTracingPipelineFeatures();
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
    score += 1000000000;
  }
  // Design a score system, key features: geometry and raytracing
  if (features.geometryShader) {
    score += 1000000;
  }
  if (ray_tracing_features.rayTracingPipeline) {
    score += 100000000;
  }
  // Consider memory size, get from existed function
  uint64_t device_local_memory_size = GetDeviceLocalMemorySize();

  // Score for memory size
  score += device_local_memory_size / 1000000;

  return score;
}

}  // namespace grassland::vulkan
