#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class RenderPass {
 public:
  RenderPass(Device *device, VkFormat color_format);
  ~RenderPass();

 private:
  VK_HANDLE(VkRenderPass)
  Device *device_{nullptr};
};
}  // namespace grassland::vulkan
