#pragma once
#include "render_pass.h"

namespace grassland::vulkan {
class Framebuffer {
 public:
  GRASSLAND_CANNOT_COPY(Framebuffer)
  Framebuffer(const RenderPass &render_pass,
              VkExtent2D extent,
              const std::vector<VkImageView> &color_attachments,
              VkImageView depth_attachment);
  ~Framebuffer();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkFramebuffer, framebuffer_)
};
}  // namespace grassland::vulkan
