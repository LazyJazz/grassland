#include <grassland/vulkan/physical_device.h>

#include "surface.h"

namespace grassland::vulkan {

PhysicalDevice::PhysicalDevice(VkPhysicalDevice handle) {
  handle_ = handle;
}

std::string PhysicalDevice::DeviceName() const {
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(handle_, &properties);
  return properties.deviceName;
}
int PhysicalDevice::HasGraphicsFamily() const {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(handle_, &queue_family_count,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queue_family_properties(
      queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(handle_, &queue_family_count,
                                           queue_family_properties.data());

  int i = 0;
  for (const auto &queue_family_property : queue_family_properties) {
    if (queue_family_property.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      return i;
    }
    i++;
  }
  return -1;
}

int PhysicalDevice::HasPresentFamily(Surface *surface) const {
  uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(handle_, &queue_family_count,
                                           nullptr);

  std::vector<VkQueueFamilyProperties> queue_family_properties(
      queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(handle_, &queue_family_count,
                                           queue_family_properties.data());
  int i = 0;
  for (const auto &queue_family_property : queue_family_properties) {
    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(handle_, i, surface->GetHandle(),
                                         &presentSupport);
    if (presentSupport) {
      return i;
    }
    i++;
  }
  return -1;
}

std::vector<PhysicalDevice> GetPhysicalDevices(Instance *instance) {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance->GetHandle(), &device_count, nullptr);
  std::vector<VkPhysicalDevice> vk_physical_devices(device_count);
  vkEnumeratePhysicalDevices(instance->GetHandle(), &device_count,
                             vk_physical_devices.data());
  std::vector<PhysicalDevice> physical_devices;

  for (auto &vk_physical_device : vk_physical_devices) {
    physical_devices.emplace_back(vk_physical_device);
  }

  return physical_devices;
}

}  // namespace grassland::vulkan
