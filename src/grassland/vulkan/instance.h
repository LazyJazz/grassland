#pragma once
#include <grassland/vulkan/util.h>
#include <vulkan/vulkan.h>

namespace grassland::vulkan {
class Instance {
 public:
  Instance();
  ~Instance();

 private:
  void CreateInstance();
  void CreateDebugMessenger();
  GRASSLAND_VULKAN_HANDLE(VkInstance)
  VkDebugUtilsMessengerEXT debug_messenger_;
};
}  // namespace grassland::vulkan
