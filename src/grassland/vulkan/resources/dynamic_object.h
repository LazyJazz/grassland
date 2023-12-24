#pragma once
#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class DynamicObject {
 public:
  virtual bool SyncData(VkCommandBuffer cmd_buffer, uint32_t frame_index) = 0;
};
}  // namespace grassland::vulkan
