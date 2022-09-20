#include <grassland/logging/logging.h>
#include <grassland/vulkan/frame_buffer.h>

namespace grassland::vulkan {

FrameBuffer::FrameBuffer(Device *device,
                         int width,
                         int height,
                         RenderPass *render_pass,
                         const std::vector<ImageView *> &image_views)
    : handle_{} {
  device_ = device;
  std::vector<VkImageView> vk_image_views(image_views.size());
  for (int i = 0; i < vk_image_views.size(); i++) {
    vk_image_views[i] = image_views[i]->GetHandle();
  }
  VkFramebufferCreateInfo framebufferInfo{};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = render_pass->GetHandle();
  framebufferInfo.attachmentCount =
      static_cast<uint32_t>(vk_image_views.size());
  framebufferInfo.pAttachments = vk_image_views.data();
  framebufferInfo.width = width;
  framebufferInfo.height = height;
  framebufferInfo.layers = 1;

  if (vkCreateFramebuffer(device_->GetHandle(), &framebufferInfo, nullptr,
                          &handle_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create framebuffer!");
  }
}

FrameBuffer::FrameBuffer(Device *device,
                         int width,
                         int height,
                         RenderPass *render_pass,
                         ImageView *color_image_view)
    : FrameBuffer(device,
                  width,
                  height,
                  render_pass,
                  std::vector<ImageView *>{color_image_view}) {
}

FrameBuffer::FrameBuffer(Device *device,
                         int width,
                         int height,
                         RenderPass *render_pass,
                         ImageView *color_image_view,
                         ImageView *depth_image_view)
    : FrameBuffer(
          device,
          width,
          height,
          render_pass,
          std::vector<ImageView *>{color_image_view, depth_image_view}) {
}

FrameBuffer::~FrameBuffer() {
  vkDestroyFramebuffer(device_->GetHandle(), GetHandle(), nullptr);
}

}  // namespace grassland::vulkan
