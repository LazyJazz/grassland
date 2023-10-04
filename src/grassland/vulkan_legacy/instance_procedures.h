#pragma once

#include "grassland/vulkan_legacy/image.h"
#include "vulkan/vulkan.h"

namespace grassland::vulkan_legacy {
class InstanceProcedures {
 public:
  static InstanceProcedures *GetStaticInstance();
  void SetInstance(VkInstance instance);
  [[nodiscard]] VkInstance GetInstance() const;
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkCreateDebugUtilsMessengerEXT);
  GRASSLAND_VULKAN_PROCEDURE_VAR(vkDestroyDebugUtilsMessengerEXT);

 private:
  VkInstance instance_{nullptr};
};
}  // namespace grassland::vulkan_legacy
