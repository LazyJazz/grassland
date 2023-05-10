#pragma once
#include "grassland/util/util.h"
#include "grassland/vulkan/vulkan_util.h"
#include "vulkan/vulkan.h"

namespace grassland::vulkan {
class Instance {
 public:
  Instance();

 private:
  GRASSLAND_VULKAN_HANDLE(VkInstance, instance_)
};
}  // namespace grassland::vulkan
