#pragma once
#include "grassland/vulkan/core/device.h"

#include <map>

namespace grassland::vulkan {

namespace {
DeviceSettings GetDefaultDeviceSettings(Instance *instance,
                                        const PhysicalDevice &physical_device,
                                        Surface *surface,
                                        bool enable_raytracing) {
  DeviceSettings settings{physical_device};
  settings.surface = surface;
  settings.enable_raytracing = false;
  return settings;
}
}  // namespace

Device::Device(Instance *instance,
               const class PhysicalDevice &physical_device,
               Surface *surface,
               bool enable_raytracing)
    : Device(instance,
             GetDefaultDeviceSettings(instance,
                                      physical_device,
                                      surface,
                                      enable_raytracing)) {
}

Device::Device(Instance *instance,
               const grassland::vulkan::DeviceSettings &settings)
    : instance_(instance),
      physical_device_(settings.physical_device),
      device_(VK_NULL_HANDLE),
      graphics_queue_(),
      single_time_command_queue_(),
      present_queue_() {
  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::map<uint32_t, uint32_t> uniqueQueueFamilies = {
      {settings.physical_device.GraphicsFamilyIndex(),
       std::min(2u, settings.physical_device
                        .GetQueueFamilyProperties()[settings.physical_device
                                                        .GraphicsFamilyIndex()]
                        .queueCount)}};

  std::vector<const char *> device_extensions;
  if (settings.surface) {
    auto present_family_index =
        settings.physical_device.PresentFamilyIndex(settings.surface);
    if (present_family_index != settings.physical_device.GraphicsFamilyIndex())
      uniqueQueueFamilies[present_family_index] = 1;
    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
#ifdef __APPLE__
  device_extensions.push_back("VK_KHR_portability_subset");
#endif
  device_extensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  // device_extensions.push_back(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);

  VkPhysicalDeviceRayQueryFeaturesKHR physical_device_ray_query_features{};
  VkPhysicalDeviceBufferDeviceAddressFeaturesEXT
      physical_device_buffer_device_address_features{};
  VkPhysicalDeviceRayTracingPipelineFeaturesKHR
      physical_device_ray_tracing_pipeline_features{};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR
      physical_device_acceleration_structure_features{};
  VkPhysicalDeviceSynchronization2FeaturesKHR
      physical_device_synchronization2_features{};
  physical_device_synchronization2_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR;
  physical_device_synchronization2_features.synchronization2 = VK_TRUE;
  physical_device_ray_query_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
  physical_device_ray_query_features.rayQuery = VK_TRUE;
  physical_device_ray_tracing_pipeline_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
  physical_device_ray_tracing_pipeline_features.rayTracingPipeline = VK_TRUE;
  physical_device_acceleration_structure_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  physical_device_acceleration_structure_features.accelerationStructure =
      VK_TRUE;
  physical_device_buffer_device_address_features.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT;
  physical_device_buffer_device_address_features.bufferDeviceAddress = VK_TRUE;

  void *p_extra_device_features = nullptr;
#define ADD_DEVICE_FEATURE(pointer, feature) \
  do {                                       \
    feature.pNext = pointer;                 \
    pointer = &feature;                      \
  } while (false)

  device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  ADD_DEVICE_FEATURE(p_extra_device_features,
                     physical_device_buffer_device_address_features);
  //  ADD_DEVICE_FEATURE(p_extra_device_features,
  //                     physical_device_synchronization2_features);
  if (settings.enable_raytracing) {
    device_extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    device_extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
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

  float queuePriority[2] = {1.0f, 1.0f};
  for (auto queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily.first;
    queueCreateInfo.queueCount = queueFamily.second;
    queueCreateInfo.pQueuePriorities = queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures =
      settings.physical_device.GetPhysicalDeviceFeatures();
  VkDeviceCreateInfo createInfo{};

  if (instance->Settings().enable_validation_layers) {
    createInfo.enabledLayerCount =
        static_cast<uint32_t>(vulkan::validationLayers.size());
    createInfo.ppEnabledLayerNames = vulkan::validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

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

  graphics_queue_ =
      Queue(this, settings.physical_device.GraphicsFamilyIndex(), 0);
  single_time_command_queue_ = Queue(
      this, settings.physical_device.GraphicsFamilyIndex(),
      uniqueQueueFamilies[settings.physical_device.GraphicsFamilyIndex()] - 1);

  if (settings.surface) {
    present_queue_ = Queue(
        this, settings.physical_device.PresentFamilyIndex(settings.surface), 0);
  }

  device_procedures_.Initialize(device_, settings.enable_raytracing);

  // Create vma allocator
  VmaAllocatorCreateInfo allocator_info{};
  allocator_info.physicalDevice = settings.physical_device.Handle();
  allocator_info.device = device_;
  allocator_info.instance = instance->Handle();
  vmaCreateAllocator(&allocator_info, &allocator_);
}

Device::~Device() {
  vmaDestroyAllocator(allocator_);
  vkDestroyDevice(device_, nullptr);
}

VkDevice Device::Handle() const {
  return device_;
}

class PhysicalDevice Device::PhysicalDevice() const {
  return physical_device_;
}

Queue Device::PresentQueue() const {
  return present_queue_;
}

Queue Device::GraphicsQueue() const {
  return graphics_queue_;
}

Queue Device::SingleTimeCommandQueue() const {
  return single_time_command_queue_;
}

void Device::WaitIdle() const {
  vkDeviceWaitIdle(device_);
}

void Device::NameObject(VkImage image, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_IMAGE;
  name_info.objectHandle = reinterpret_cast<uint64_t>(image);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkBuffer buffer, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_BUFFER;
  name_info.objectHandle = reinterpret_cast<uint64_t>(buffer);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkImageView image_view, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_IMAGE_VIEW;
  name_info.objectHandle = reinterpret_cast<uint64_t>(image_view);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkDeviceMemory memory, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_DEVICE_MEMORY;
  name_info.objectHandle = reinterpret_cast<uint64_t>(memory);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkCommandPool command_pool, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_COMMAND_POOL;
  name_info.objectHandle = reinterpret_cast<uint64_t>(command_pool);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkShaderModule shader_module, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_SHADER_MODULE;
  name_info.objectHandle = reinterpret_cast<uint64_t>(shader_module);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkAccelerationStructureKHR acceleration_structure,
                        const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR;
  name_info.objectHandle = reinterpret_cast<uint64_t>(acceleration_structure);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkPipeline pipeline, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_PIPELINE;
  name_info.objectHandle = reinterpret_cast<uint64_t>(pipeline);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkPipelineLayout pipeline_layout,
                        const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_PIPELINE_LAYOUT;
  name_info.objectHandle = reinterpret_cast<uint64_t>(pipeline_layout);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkDescriptorSetLayout descriptor_set_layout,
                        const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT;
  name_info.objectHandle = reinterpret_cast<uint64_t>(descriptor_set_layout);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkDescriptorPool descriptor_pool,
                        const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_DESCRIPTOR_POOL;
  name_info.objectHandle = reinterpret_cast<uint64_t>(descriptor_pool);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkDescriptorSet descriptor_set,
                        const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_DESCRIPTOR_SET;
  name_info.objectHandle = reinterpret_cast<uint64_t>(descriptor_set);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkCommandBuffer command_buffer,
                        const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_COMMAND_BUFFER;
  name_info.objectHandle = reinterpret_cast<uint64_t>(command_buffer);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkFramebuffer framebuffer, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_FRAMEBUFFER;
  name_info.objectHandle = reinterpret_cast<uint64_t>(framebuffer);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkRenderPass render_pass, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_RENDER_PASS;
  name_info.objectHandle = reinterpret_cast<uint64_t>(render_pass);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

void Device::NameObject(VkSampler sampler, const std::string &name) {
  VkDebugUtilsObjectNameInfoEXT name_info{};
  name_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
  name_info.objectType = VK_OBJECT_TYPE_SAMPLER;
  name_info.objectHandle = reinterpret_cast<uint64_t>(sampler);
  name_info.pObjectName = name.c_str();
  instance_->Procedures().vkSetDebugUtilsObjectNameEXT(device_, &name_info);
}

}  // namespace grassland::vulkan
