#pragma once

#include <grassland/vulkan/image_view.h>
#include <grassland/vulkan/render_pass.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Framebuffer {
 public:
  Framebuffer(Device *device,
              int width,
              int height,
              RenderPass *render_pass,
              ImageView *color_image_view);
  Framebuffer(Device *device,
              int width,
              int height,
              RenderPass *render_pass,
              ImageView *color_image_view,
              ImageView *depth_image_view);
  Framebuffer(Device *device,
              int width,
              int height,
              RenderPass *render_pass,
              const std::vector<ImageView *> &image_views);
  ~Framebuffer();

 private:
  GRASSLAND_VULKAN_HANDLE(VkFramebuffer)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
