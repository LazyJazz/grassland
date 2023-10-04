#include "grassland/vulkan/core/instance.h"

#include <iostream>

#include "GLFW/glfw3.h"
#include "grassland/vulkan/core/instance_procedures.h"
#include "grassland/vulkan/core/physical_device.h"

namespace grassland::vulkan {

namespace {
InstanceSettings GetDefaultInstanceSettings() {
  InstanceSettings settings;
  settings.EnableSurfaceSupport();
  if (kDefaultEnableValidationLayers) {
    settings.EnableValidationLayers();
  }
  return settings;
}

bool CheckValidationLayerSupport() {
  uint32_t layer_count;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

  std::vector<VkLayerProperties> available_layers(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

  for (const char *layerName : validationLayers) {
    bool layer_found = false;

    for (const auto &layerProperties : available_layers) {
      if (std::strcmp(layerName, layerProperties.layerName) == 0) {
        layer_found = true;
        break;
      }
    }

    if (!layer_found) {
      return false;
    }
  }

  return true;
}

VKAPI_ATTR VkBool32 VKAPI_CALL DebugUtilsMessengerUserCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
    VkDebugUtilsMessageTypeFlagsEXT message_type,
    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
    void *user_data) {
  if (message_severity <= VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    return VK_FALSE;
  }
  std::string message_tag;
  auto add_tag = [&message_tag](const char *tag) {
    if (!message_tag.empty()) {
      message_tag += ", ";
    }
    message_tag += tag;
  };
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
    add_tag("ERROR");
  }
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    add_tag("WARNING");
  }
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    add_tag("INFO");
  }
  if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT) {
    add_tag("VERBOSE");
  }
  std::cerr << fmt::format("validation layer ({}): {}", message_tag,
                           callback_data->pMessage)
            << std::endl;
  return VK_FALSE;
}

void GenerateDebugMessengerCreateInfo(
    VkDebugUtilsMessengerCreateInfoEXT &create_info) {
  create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
  create_info.messageSeverity =
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
      VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
  create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
  create_info.pfnUserCallback = DebugUtilsMessengerUserCallback;
}

}  // namespace

Instance::~Instance() {
  if (settings_.enable_validation_layers) {
    instance_procedures_.vkDestroyDebugUtilsMessengerEXT(
        instance_, debug_messenger_, nullptr);
  }
  // Destroy Vulkan instance
  vkDestroyInstance(instance_, nullptr);
}

VkInstance Instance::Handle() const {
  return instance_;
}

void Instance::EnumeratePhysicalDevices(
    std::vector<PhysicalDevice> &physical_devices) const {
  // Enumerate physical devices
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
  std::vector<VkPhysicalDevice> devices(device_count);
  vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());
  for (const auto &device : devices) {
    physical_devices.emplace_back(device);
  }
}

Instance::Instance() : Instance(GetDefaultInstanceSettings()) {
}

Instance::Instance(InstanceSettings settings) {
  VkInstanceCreateInfo create_info{};
  VkDebugUtilsMessengerCreateInfoEXT debug_create_info{};

  // Check validation layer support and enable validation layers
  if (settings.enable_validation_layers) {
    if (!CheckValidationLayerSupport()) {
      LAND_ERROR("[Vulkan] validation layer is required, but not supported.");
    }
    settings.extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    create_info.enabledLayerCount =
        static_cast<uint32_t>(vulkan::validationLayers.size());
    create_info.ppEnabledLayerNames = vulkan::validationLayers.data();

    GenerateDebugMessengerCreateInfo(debug_create_info);
    create_info.pNext = &debug_create_info;
  } else {
    create_info.enabledLayerCount = 0;
    create_info.pNext = nullptr;
  }

  // Enable portable extension on apple devices
#ifdef __APPLE__
  settings.extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  // Create Vulkan instance
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &settings.app_info;
  create_info.enabledExtensionCount =
      static_cast<uint32_t>(settings.extensions.size());
  create_info.ppEnabledExtensionNames = settings.extensions.data();

  // Set portable extension flag on apple devices
#ifdef __APPLE__
  create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create instance!");
  }

  instance_procedures_.Initialize(instance_);

  if (settings.enable_validation_layers) {
    if (instance_procedures_.vkCreateDebugUtilsMessengerEXT(
            instance_, &debug_create_info, nullptr, &debug_messenger_) !=
        VK_SUCCESS) {
      LAND_ERROR("[Vulkan] failed to set up debug messenger!");
    }
  }

  settings_ = settings;
}

const InstanceSettings &Instance::Settings() const {
  return settings_;
}
}  // namespace grassland::vulkan
