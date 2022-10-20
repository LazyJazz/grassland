#include <grassland/util/logging.h>
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/queue.h>

#include <set>

namespace grassland::vulkan {

Device::Device(PhysicalDevice *physical_device)
    : Device(physical_device, nullptr) {
}

Device::Device(PhysicalDevice *physical_device, Surface *surface) : handle_{} {
  physical_device_ = physical_device;
  surface_ = surface;
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {
      physical_device->GraphicsFamilyIndex()};

  std::vector<const char *> device_extensions;
  if (surface) {
    uniqueQueueFamilies.insert(physical_device->PresentFamilyIndex(surface));
    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
#ifdef __APPLE__
  device_extensions.push_back("VK_KHR_portability_subset");
#endif

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }
  VkPhysicalDeviceFeatures deviceFeatures = physical_device_->GetFeatures();
  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount =
      static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.pEnabledFeatures = &deviceFeatures;
  createInfo.enabledExtensionCount =
      static_cast<uint32_t>(device_extensions.size());
  if (createInfo.enabledExtensionCount) {
    createInfo.ppEnabledExtensionNames = device_extensions.data();
  }

  if (kEnableValidationLayers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  if (vkCreateDevice(physical_device->GetHandle(), &createInfo, nullptr,
                     &handle_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create logical device!");
  }

  graphics_queue_ = new Queue(this, physical_device_->GraphicsFamilyIndex());
}

Device::~Device() {
  delete graphics_queue_;
  vkDestroyDevice(handle_, nullptr);
}

PhysicalDevice *Device::GetPhysicalDevice() {
  return physical_device_;
}

Surface *Device::GetSurface() {
  return surface_;
}

Queue *Device::GetGraphicsQueue() {
  return graphics_queue_;
}

void Device::WaitIdle() {
  vkDeviceWaitIdle(handle_);
}

}  // namespace grassland::vulkan
