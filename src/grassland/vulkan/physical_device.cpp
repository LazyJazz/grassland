#include <grassland/logging/logging.h>
#include <grassland/vulkan/physical_device.h>
#include <grassland/vulkan/surface.h>

#include <utility>

namespace grassland::vulkan {

PhysicalDevice::PhysicalDevice(VkPhysicalDevice handle) {
  handle_ = handle;
  vkGetPhysicalDeviceProperties(handle_, &properties_);
  vkGetPhysicalDeviceFeatures(handle_, &features_);
}

std::string PhysicalDevice::DeviceName() const {
  return properties_.deviceName;
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

bool PhysicalDevice::HasPresentationSupport() const {
  return false;
}

void PhysicalDevice::PrintDeviceProperties() const {
  spdlog::info("  {}", properties_.deviceName);
  spdlog::info("    Vendor ID  : {:#X}", properties_.vendorID);
  spdlog::info("    Device Type: {}", [](VkPhysicalDeviceType device_type) {
    if (device_type == VK_PHYSICAL_DEVICE_TYPE_CPU)
      return "CPU";
    else if (device_type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
      return "Discrete";
    else if (device_type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
      return "Integrated";
    else if (device_type == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)
      return "Virtual";
    else if (device_type == VK_PHYSICAL_DEVICE_TYPE_OTHER)
      return "Other";
    return "Unknown";
  }(properties_.deviceType));
}

void PhysicalDevice::PrintDeviceFeatures() const {
  spdlog::info("Geometry shader: {}",
               features_.geometryShader ? "True" : "False");
}

bool PhysicalDevice::IsDiscreteGPU() const {
  return properties_.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

bool PhysicalDevice::HasGeometryShader() const {
  return features_.geometryShader;
}

VkPhysicalDeviceFeatures PhysicalDevice::GetFeatures() const {
  return features_;
}

VkPhysicalDeviceProperties PhysicalDevice::GetProperties() const {
  return properties_;
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

PhysicalDevice vulkan::PickPhysicalDevice(
    const std::vector<PhysicalDevice> &device_list,
    const std::function<int(PhysicalDevice)> &rate_function) {
  if (device_list.empty()) {
    LAND_ERROR("Vulkan: No device found!");
  }
  PhysicalDevice result = device_list[0];
  int res_score = rate_function(result);
  for (size_t i = 1; i < device_list.size(); i++) {
    int score = rate_function(device_list[i]);
    if (score > res_score) {
      res_score = score;
      result = device_list[i];
    }
  }
  return result;
}

PhysicalDevice PickPhysicalDevice(
    Instance *instance,
    const std::function<int(PhysicalDevice)> &rate_function) {
  return PickPhysicalDevice(GetPhysicalDevices(instance), rate_function);
}

}  // namespace grassland::vulkan
