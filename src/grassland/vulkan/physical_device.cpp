#include "physical_device.h"

namespace grassland::vulkan {

PhysicalDevice::PhysicalDevice(VkPhysicalDevice physical_device)
    : physical_device_(physical_device) {
}

std::string PhysicalDevice::DeviceName() const {
  return GetProperties().deviceName;
}

VkPhysicalDeviceProperties PhysicalDevice::GetProperties() const {
  VkPhysicalDeviceProperties properties{};
  vkGetPhysicalDeviceProperties(physical_device_, &properties);
  return properties;
}

VkPhysicalDeviceFeatures PhysicalDevice::GetFeatures() const {
  VkPhysicalDeviceFeatures features{};
  vkGetPhysicalDeviceFeatures(physical_device_, &features);
  return features;
}

VkPhysicalDeviceRayTracingPipelinePropertiesKHR
PhysicalDevice::GetRayTracingProperties() const {
  VkPhysicalDeviceProperties2 properties2{};
  VkPhysicalDeviceRayTracingPipelinePropertiesKHR ray_tracing_properties{};
  properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties2.pNext = &ray_tracing_properties;
  ray_tracing_properties.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
  vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
  return ray_tracing_properties;
}

VkPhysicalDeviceRayTracingPipelineFeaturesKHR
PhysicalDevice::GetRayTracingFeatures() const {
  VkPhysicalDeviceFeatures2 features2{};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR ray_tracing_features{};
  features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features2.pNext = &ray_tracing_features;
  ray_tracing_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  vkGetPhysicalDeviceFeatures2(physical_device_, &features2);
  return ray_tracing_features;
}

}  // namespace grassland::vulkan
