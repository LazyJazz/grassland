#include "grassland/vulkan/core/instance_settings.h"

#include "GLFW/glfw3.h"

namespace grassland::vulkan {

InstanceSettings::InstanceSettings() {
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = "Grassland";
  app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.pEngineName = "Grassland Engine";
  app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  app_info.apiVersion = VK_API_VERSION_1_2;

  extensions = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
}

void InstanceSettings::EnableSurfaceSupport() {
  // Get required extensions from glfw API
  uint32_t glfw_extension_count = 0;
  const char **glfw_extensions;
  glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
  for (uint32_t i = 0; i < glfw_extension_count; i++) {
    extensions.push_back(glfw_extensions[i]);
  }
}

void InstanceSettings::EnableValidationLayers() {
  enable_validation_layers = true;
}
}  // namespace grassland::vulkan
