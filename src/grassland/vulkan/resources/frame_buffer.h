#pragma once

#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class Framebuffer {
 public:
  Framebuffer(class Core *core,
              VkExtent2D extent,
              VkRenderPass render_pass,
              const std::vector<VkImageView> &image_views);
  ~Framebuffer();

  [[nodiscard]] VkFramebuffer Handle() const;
  [[nodiscard]] class Core *Core() const;

  [[nodiscard]] VkExtent2D Extent() const;

 private:
  class Core *core_{};
  VkFramebuffer framebuffer_{};
  VkExtent2D extent_{};
};
}  // namespace grassland::vulkan
