#include "device.h"

#include "enumerate.h"
#include "set"

namespace grassland::vulkan {

namespace {
void CheckRequiredExtensions(
    VkPhysicalDevice physical_device,
    const std::vector<const char *> &required_extensions) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count,
                                       nullptr);
  std::vector<VkExtensionProperties> available_extensions(count);
  vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count,
                                       available_extensions.data());

  std::set<std::string> required(required_extensions.begin(),
                                 required_extensions.end());
  for (auto &extension : available_extensions) {
    required.erase(extension.extensionName);
  }

  if (!required.empty()) {
    std::string missing_extension_list;
    for (const auto &extension : required) {
      missing_extension_list += '\n';
      missing_extension_list += extension;
    }

    LAND_ERROR("\nMissing required extension:{}", missing_extension_list);
  }
}

std::vector<VkQueueFamilyProperties>::const_iterator FindQueue(
    const std::vector<VkQueueFamilyProperties> &queue_families,
    const std::string &name,
    const VkQueueFlags required_bits,
    const VkQueueFlags excluded_bits) {
  const auto family =
      std::find_if(queue_families.begin(), queue_families.end(),
                   [required_bits,
                    excluded_bits](const VkQueueFamilyProperties &queueFamily) {
                     return queueFamily.queueCount > 0 &&
                            queueFamily.queueFlags & required_bits &&
                            !(queueFamily.queueFlags & excluded_bits);
                   });

  if (family == queue_families.end()) {
    LAND_ERROR("found no matching {} queue", name);
  }

  return family;
}
}  // namespace

Device::Device(class PhysicalDevice physical_device,
               class Surface *surface,
               bool enable_validation_layer,
               const std::vector<const char *> &extra_extensions,
               void *extra_features)
    : physical_device_(physical_device), surface_(surface) {
  std::vector<const char *> device_extensions{};

#ifdef __APPLE__
  device_extensions.push_back("VK_KHR_portability_subset");
#endif
  device_extensions.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
  device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  if (surface_) {
    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  }
  if (!extra_extensions.empty()) {
    device_extensions.insert(device_extensions.end(), extra_extensions.begin(),
                             extra_extensions.end());
  }

  CheckRequiredExtensions(physical_device_.Handle(), device_extensions);

  spdlog::info("Creating Device ({})...", physical_device_.DeviceName());

  spdlog::info("Device extensions:");
  for (auto extension : device_extensions) {
    spdlog::info("- {}", extension);
  }
  spdlog::info("");

  const auto queue_families = GetEnumerateVector(
      physical_device_.Handle(), vkGetPhysicalDeviceQueueFamilyProperties);
  const auto graphics_family =
      FindQueue(queue_families, "graphics", VK_QUEUE_GRAPHICS_BIT, 0);
  const auto compute_family = FindQueue(
      queue_families, "compute", VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT);

  if (surface_) {
    const auto present_family = std::find_if(
        queue_families.begin(), queue_families.end(),
        [&](const VkQueueFamilyProperties &queueFamily) {
          VkBool32 presentSupport = false;
          const uint32_t i =
              static_cast<uint32_t>(&*queue_families.cbegin() - &queueFamily);
          vkGetPhysicalDeviceSurfaceSupportKHR(physical_device_.Handle(), i,
                                               surface_->Handle(),
                                               &presentSupport);
          return queueFamily.queueCount > 0 && presentSupport;
        });
    present_family_index_ = present_family - queue_families.begin();
  }
  graphics_family_index_ = graphics_family - queue_families.begin();
  compute_family_index_ = compute_family - queue_families.begin();

  std::set<uint32_t> unique_queue_families = {graphics_family_index_,
                                              compute_family_index_};

  if (surface_) {
    unique_queue_families.insert(present_family_index_);
  }

  float queue_priority = 1.0f;
  std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
  for (auto queue_family_index : unique_queue_families) {
    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = queue_family_index;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;
    queue_create_infos.push_back(queue_create_info);
  }

  VkPhysicalDeviceFeatures device_features = physical_device_.GetFeatures();

  VkDeviceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create_info.pNext = extra_features;
  create_info.queueCreateInfoCount =
      static_cast<uint32_t>(queue_create_infos.size());
  create_info.pQueueCreateInfos = queue_create_infos.data();
  create_info.pEnabledFeatures = &device_features;
  if (enable_validation_layer) {
    create_info.enabledLayerCount = validationLayers.size();
    create_info.ppEnabledLayerNames = validationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;
    create_info.ppEnabledLayerNames = nullptr;
  }
  create_info.enabledExtensionCount = device_extensions.size();
  create_info.ppEnabledExtensionNames = device_extensions.data();

  GRASSLAND_VULKAN_CHECK(vkCreateDevice(physical_device_.Handle(), &create_info,
                                        nullptr, &device_));

  vkGetDeviceQueue(device_, graphics_family_index_, 0, &graphics_queue_);
  vkGetDeviceQueue(device_, compute_family_index_, 0, &compute_queue_);

  if (surface_) {
    vkGetDeviceQueue(device_, present_family_index_, 0, &present_queue_);
  }

  vkGetPhysicalDeviceMemoryProperties(physical_device_.Handle(),
                                      &memory_properties_);
}

Device::~Device() {
  vkDestroyDevice(device_, nullptr);
}

}  // namespace grassland::vulkan
