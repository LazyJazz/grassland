#pragma once
#include "device.h"

namespace grassland::vulkan {
class RenderPass {
 public:
  GRASSLAND_CANNOT_COPY(RenderPass)
  RenderPass(const class Device &device,
             VkFormat color_format,
             VkFormat depth_format = VK_FORMAT_UNDEFINED);
  RenderPass(const class Device &device,
             const std::vector<VkFormat> &color_formats,
             VkFormat depth_format = VK_FORMAT_UNDEFINED);
  ~RenderPass();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkRenderPass, render_pass_)
};
}  // namespace grassland::vulkan
