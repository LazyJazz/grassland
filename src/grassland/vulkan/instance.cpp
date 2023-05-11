#include "instance.h"

#include "enumerate.h"
#include "iostream"
#include "physical_device.h"

namespace grassland::vulkan {

namespace {

std::vector<const char *> GetRequiredExtensions(
    const InstanceSettings &settings) {
  uint32_t glfw_extension_count = 0;
  const char **glfw_extensions = nullptr;
  if (settings.glfw_surface) {
    glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  }

  std::vector<const char *> extensions;

  for (uint32_t i = 0; i < glfw_extension_count; i++) {
    extensions.push_back(glfw_extensions[i]);
  }

  if (settings.validation_layer) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  return extensions;
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

const char *ObjectTypeToString(const VkObjectType objectType) {
  switch (objectType) {
#define STR(e)             \
  case VK_OBJECT_TYPE_##e: \
    return #e
    STR(UNKNOWN);
    STR(INSTANCE);
    STR(PHYSICAL_DEVICE);
    STR(DEVICE);
    STR(QUEUE);
    STR(SEMAPHORE);
    STR(COMMAND_BUFFER);
    STR(FENCE);
    STR(DEVICE_MEMORY);
    STR(BUFFER);
    STR(IMAGE);
    STR(EVENT);
    STR(QUERY_POOL);
    STR(BUFFER_VIEW);
    STR(IMAGE_VIEW);
    STR(SHADER_MODULE);
    STR(PIPELINE_CACHE);
    STR(PIPELINE_LAYOUT);
    STR(RENDER_PASS);
    STR(PIPELINE);
    STR(DESCRIPTOR_SET_LAYOUT);
    STR(SAMPLER);
    STR(DESCRIPTOR_POOL);
    STR(DESCRIPTOR_SET);
    STR(FRAMEBUFFER);
    STR(COMMAND_POOL);
    STR(SAMPLER_YCBCR_CONVERSION);
    STR(DESCRIPTOR_UPDATE_TEMPLATE);
    STR(SURFACE_KHR);
    STR(SWAPCHAIN_KHR);
    STR(DISPLAY_KHR);
    STR(DISPLAY_MODE_KHR);
    STR(DEBUG_REPORT_CALLBACK_EXT);
    STR(DEBUG_UTILS_MESSENGER_EXT);
    STR(ACCELERATION_STRUCTURE_KHR);
    STR(VALIDATION_CACHE_EXT);
    STR(PERFORMANCE_CONFIGURATION_INTEL);
    STR(DEFERRED_OPERATION_KHR);
    STR(INDIRECT_COMMANDS_LAYOUT_NV);
#undef STR
    default:
      return "unknown";
  }
}

VKAPI_ATTR VkBool32 VKAPI_CALL
VulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                    VkDebugUtilsMessageTypeFlagsEXT message_type,
                    const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                    void *user_data) {
  switch (message_severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      std::cerr << "VERBOSE: ";
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      std::cerr << "INFO: ";
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      std::cerr << "WARNING: ";
      break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      std::cerr << "ERROR: ";
      break;
    default:;
      std::cerr << "UNKNOWN: ";
  }

  switch (message_type) {
    case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
      std::cerr << "GENERAL: ";
      break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
      std::cerr << "VALIDATION: ";
      break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
      std::cerr << "PERFORMANCE: ";
      break;
    default:
      std::cerr << "UNKNOWN: ";
  }

  std::cerr << callback_data->pMessage;

  if (callback_data->objectCount > 0 &&
      message_severity > VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
    std::cerr << "\n\n  Objects (" << callback_data->objectCount << "):\n";

    for (uint32_t i = 0; i != callback_data->objectCount; ++i) {
      const auto object = callback_data->pObjects[i];
      std::cerr << "  - Object[" << i << "]: "
                << "Type: " << ObjectTypeToString(object.objectType) << ", "
                << "Handle: " << reinterpret_cast<void *>(object.objectHandle)
                << ", "
                << "Name: '" << (object.pObjectName ? object.pObjectName : "")
                << "'"
                << "\n";
    }
  }

  std::cerr << std::endl;

  return VK_FALSE;
}

template <class FuncTy>
FuncTy GetProcedure(VkInstance instance, const char *function_name) {
  auto func = (FuncTy)vkGetInstanceProcAddr(instance, function_name);
  return func;
};

#define GET_PROCEDURE(instance, function_name) \
  function_name##_ = GetProcedure<PFN_##function_name>(instance, #function_name)
}  // namespace

Instance::Instance(const InstanceSettings &settings) {
  if (settings.validation_layer && !CheckValidationLayerSupport()) {
    LAND_ERROR("[Vulkan] validation layer is required, but not supported.");
  }

  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Grassland";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "No Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  auto extensions = GetRequiredExtensions(settings);
#ifdef __APPLE__
  extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

  spdlog::info("Instance extensions:");
  for (auto extension : extensions) {
    spdlog::info("- {}", extension);
  }
  spdlog::info("");

  create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  create_info.ppEnabledExtensionNames = extensions.data();

  if (settings.validation_layer) {
    create_info.enabledLayerCount =
        static_cast<uint32_t>(validationLayers.size());
    create_info.ppEnabledLayerNames = validationLayers.data();
  } else {
    create_info.enabledLayerCount = 0;

    create_info.pNext = nullptr;
  }

#ifdef __APPLE__
  create_info.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

  GRASSLAND_VULKAN_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));

  GET_PROCEDURE(instance_, vkCreateDebugUtilsMessengerEXT);
  GET_PROCEDURE(instance_, vkDestroyDebugUtilsMessengerEXT);

  if (settings.validation_layer) {
    VkDebugUtilsMessengerCreateInfoEXT createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = VulkanDebugCallback;
    createInfo.pUserData = nullptr;

    GRASSLAND_VULKAN_CHECK(vkCreateDebugUtilsMessengerEXT_(
        instance_, &createInfo, nullptr, &messenger_));
  }
}

std::vector<PhysicalDevice> Instance::GetEnumeratePhysicalDevices() const {
  auto physical_devices =
      GetEnumerateVector(instance_, vkEnumeratePhysicalDevices);
  if (physical_devices.empty()) {
    LAND_ERROR("No Vulkan device found.");
  }
  return {physical_devices.begin(), physical_devices.end()};
}

PhysicalDevice Instance::PickDevice(bool ray_tracing) const {
  auto physical_devices = GetEnumeratePhysicalDevices();
  auto result = physical_devices[0];
  for (int i = 1; i < physical_devices.size(); i++) {
    auto current_device = physical_devices[i];
    if (ray_tracing) {
      if (current_device.GetRayTracingFeatures().rayTracingPipeline &&
          !result.GetRayTracingFeatures().rayTracingPipeline) {
        result = current_device;
      } else {
        if (current_device.GetProperties().deviceType ==
                VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
            result.GetProperties().deviceType !=
                VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
          result = current_device;
        }
      }
    } else {
      if (current_device.GetProperties().deviceType ==
              VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
          result.GetProperties().deviceType !=
              VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        result = current_device;
      } else {
        if (current_device.GetRayTracingFeatures().rayTracingPipeline &&
            !result.GetRayTracingFeatures().rayTracingPipeline) {
          result = current_device;
        }
      }
    }
  }
  if (ray_tracing && !result.GetRayTracingFeatures().rayTracingPipeline) {
    LAND_ERROR("No Ray Tracing device found.");
  }
  return result;
}

Instance::~Instance() {
  if (instance_ != VK_NULL_HANDLE) {
    if (messenger_ != VK_NULL_HANDLE) {
      vkDestroyDebugUtilsMessengerEXT_(instance_, messenger_, nullptr);
    }
    vkDestroyInstance(instance_, nullptr);
  }
}

}  // namespace grassland::vulkan
