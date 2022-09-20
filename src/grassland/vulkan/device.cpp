#include <grassland/logging/logging.h>
#include <grassland/vulkan/device.h>

namespace grassland::vulkan {

Device::Device(PhysicalDevice *physical_device) {
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = physical_device->GraphicsFamilyIndex();
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1.0f;
  queueCreateInfo.pQueuePriorities = &queuePriority;
  VkPhysicalDeviceFeatures deviceFeatures{};
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.pQueueCreateInfos = &queueCreateInfo;
  createInfo.queueCreateInfoCount = 1;

  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount = 0;

  if (kEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  if (vkCreateDevice(physical_device->GetHandle(), &createInfo, nullptr,
                     &handle_) != VK_SUCCESS) {
    LAND_ERROR("Vulkan failed to create logical device!");
  }
}

Device::~Device() {
  vkDestroyDevice(handle_, nullptr);
}

}  // namespace grassland::vulkan
