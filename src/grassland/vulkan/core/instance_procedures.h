#pragma once

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan {
class InstanceProcedures {
 public:
  InstanceProcedures() = default;
  void GetFunctionPointers(VkInstance instance);
  VkResult vkCreateDebugUtilsMessengerEXT(
      VkInstance instance,
      const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo,
      const VkAllocationCallbacks *pAllocator,
      VkDebugUtilsMessengerEXT *pMessenger);
  void vkDestroyDebugUtilsMessengerEXT(VkInstance instance,
                                       VkDebugUtilsMessengerEXT messenger,
                                       const VkAllocationCallbacks *pAllocator);

 private:
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateDebugUtilsMessengerEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyDebugUtilsMessengerEXT);
};
}  // namespace grassland::vulkan
