#pragma once
#include <vulkan/vulkan.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class Instance {
 public:
  explicit Instance(
      bool require_surface = true,
      bool enable_validation_layers = vulkan::kDefaultEnableValidationLayers);
  ~Instance();

 private:
  void CreateInstance();
  void CreateDebugMessenger();
  GRASSLAND_VULKAN_HANDLE(VkInstance)
  VkDebugUtilsMessengerEXT debug_messenger_{};
  bool require_surface_{true};
  bool enable_validation_layers_{true};
};
}  // namespace grassland::vulkan_legacy
