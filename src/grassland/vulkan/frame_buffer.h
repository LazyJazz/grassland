#pragma once

#include <grassland/vulkan/image_view.h>
#include <grassland/vulkan/render_pass.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class FrameBuffer {
 public:
  FrameBuffer(Device *device,
              int width,
              int height,
              RenderPass *render_pass,
              ImageView *color_image_view);
  FrameBuffer(Device *device,
              int width,
              int height,
              RenderPass *render_pass,
              ImageView *color_image_view,
              ImageView *depth_image_view);
  FrameBuffer(Device *device,
              int width,
              int height,
              RenderPass *render_pass,
              const std::vector<ImageView *> &image_views);
  ~FrameBuffer();

 private:
  VK_HANDLE(VkFramebuffer)
  VK_DEVICE_PTR
};
}  // namespace grassland::vulkan
