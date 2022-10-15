#pragma once
#include <grassland/vulkan/util.h>
#include <vulkan/vulkan.h>

namespace grassland::vulkan {
class Instance {
 public:
  Instance(bool require_surface = true);
  ~Instance();

 private:
  void CreateInstance();
  void CreateDebugMessenger();
  GRASSLAND_VULKAN_HANDLE(VkInstance)
  VkDebugUtilsMessengerEXT debug_messenger_;
  bool require_surface_;
};
}  // namespace grassland::vulkan
