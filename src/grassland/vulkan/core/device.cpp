#pragma once
#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {

namespace {
DeviceSettings GetDefaultDeviceSettings(Instance *instance,
                                        const PhysicalDevice &physical_device,
                                        Surface *surface,
                                        bool enable_raytracing) {
  DeviceSettings settings{physical_device};
  settings.surface = surface;
  settings.enable_raytracing = true;
  return settings;
}
}  // namespace

Device::Device(Instance *instance,
               const PhysicalDevice &physical_device,
               Surface *surface,
               bool enable_raytracing)
    : Device(instance,
             GetDefaultDeviceSettings(instance,
                                      physical_device,
                                      surface,
                                      enable_raytracing)) {
}

Device::Device(Instance *instance,
               const grassland::vulkan::DeviceSettings &settings) {
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = {
      settings.physical_device.GraphicsFamilyIndex()};

  std::vector<const char *> device_extensions;
  if (settings.surface) {
    uniqueQueueFamilies.insert(
        settings.physical_device.PresentFamilyIndex(settings.surface));
    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
#ifdef __APPLE__
  device_extensions.push_back("VK_KHR_portability_subset");
#endif
  device_extensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

  VkPhysicalDeviceRayQueryFeaturesKHR physical_device_ray_query_features{};
  VkPhysicalDeviceBufferDeviceAddressFeaturesEXT
      physical_device_buffer_device_address_features{};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR
      physical_device_ray_tracing_pipeline_features{};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR
      physical_device_acceleration_structure_features{};
  physical_device_ray_query_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
  physical_device_ray_query_features.rayQuery = VK_TRUE;
  physical_device_ray_tracing_pipeline_features.pNext =
      &physical_device_ray_query_features;
  physical_device_ray_tracing_pipeline_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  physical_device_ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
  physical_device_acceleration_structure_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  physical_device_acceleration_structure_features.accelerationStructure =
      VK_TRUE;
  physical_device_acceleration_structure_features.pNext =
      &physical_device_ray_tracing_pipeline_features;
  physical_device_buffer_device_address_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT;
  physical_device_buffer_device_address_features.bufferDeviceAddress = VK_TRUE;
  physical_device_buffer_device_address_features.pNext =
      &physical_device_acceleration_structure_features;

  void *p_extra_device_features = nullptr;
#define ADD_DEVICE_FEATURE(pointer, feature) \
  do {                                       \
    feature.pNext = pointer;                 \
    pointer = &feature;                      \
  } while (false)

  ADD_DEVICE_FEATURE(p_extra_device_features,
                     physical_device_buffer_device_address_features);
  if (settings.enable_raytracing) {
    device_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);

    // Add ray tracing features
    ADD_DEVICE_FEATURE(p_extra_device_features,
                       physical_device_ray_tracing_pipeline_features);
    ADD_DEVICE_FEATURE(p_extra_device_features,
                       physical_device_acceleration_structure_features);
    ADD_DEVICE_FEATURE(p_extra_device_features,
                       physical_device_ray_query_features);
  }

  spdlog::info(
      "--- {} ---",
      settings.physical_device.GetPhysicalDeviceProperties().deviceName);
  spdlog::info(
      "Device vendor: {}",
      PCIVendorIDToName(
          settings.physical_device.GetPhysicalDeviceProperties().vendorID));
  spdlog::info("Device extensions:");
  for (auto extension : device_extensions) {
    spdlog::info("- {}", extension);
  }
  spdlog::info("");

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures =
      settings.physical_device.GetPhysicalDeviceFeatures();
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

  if (instance->Settings().enable_validation_layers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(vulkan::validationLayers.size());
    createInfo.ppEnabledLayerNames = vulkan::validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  VkPhysicalDeviceDescriptorIndexingFeaturesEXT
      physicalDeviceDescriptorIndexingFeatures{};
  physicalDeviceDescriptorIndexingFeatures.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT;
  physicalDeviceDescriptorIndexingFeatures
      .shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
  physicalDeviceDescriptorIndexingFeatures.runtimeDescriptorArray = VK_TRUE;
  physicalDeviceDescriptorIndexingFeatures
      .descriptorBindingVariableDescriptorCount = VK_TRUE;

  physicalDeviceDescriptorIndexingFeatures.pNext = p_extra_device_features;
  createInfo.pNext = &physicalDeviceDescriptorIndexingFeatures;

  if (vkCreateDevice(settings.physical_device.Handle(), &createInfo, nullptr,
                     &device_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create logical device!");
  }

  graphics_queue_ = Queue(this, settings.physical_device.GraphicsFamilyIndex());

  if (settings.surface) {
    present_queue_ = Queue(
        this, settings.physical_device.PresentFamilyIndex(settings.surface));
  }

  device_procedures_.Initialize(device_);
}

Device::~Device() {
  vkDestroyDevice(device_, nullptr);
}

VkDevice Device::Handle() const {
  return device_;
}

Queue Device::PresentQueue() const {
  return present_queue_;
}

Queue Device::GraphicsQueue() const {
  return graphics_queue_;
}
}  // namespace grassland::vulkan
