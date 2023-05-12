#pragma once
#include "device.h"

namespace grassland::vulkan {
class RenderPass {
 public:
  GRASSLAND_CANNOT_COPY(RenderPass)
 private:
  GRASSLAND_VULKAN_HANDLE(VkRenderPass, render_pass_)
};
}  // namespace grassland::vulkan
