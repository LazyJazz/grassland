#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class RenderPass {
 public:
  RenderPass(Device *device, VkFormat color_format);
  ~RenderPass();

 private:
  GRASSLAND_VULKAN_HANDLE(VkRenderPass)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
