#pragma once
#include <grassland/vulkan/util.h>
#include <vulkan/vulkan.h>

namespace grassland::vulkan {
class Instance {
 public:
  Instance();
  ~Instance();

 private:
  VK_HANDLE(VkInstance)
};
}  // namespace grassland::vulkan
