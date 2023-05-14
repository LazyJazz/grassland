#include "framebuffer.h"

namespace grassland::vulkan {

Framebuffer::Framebuffer(const RenderPass &render_pass,
                         VkExtent2D extent,
                         const std::vector<VkImageView> &color_attachments,
                         VkImageView depth_attachment)
    : device_(render_pass.Device()) {
  auto attachments = color_attachments;
  attachments.push_back(depth_attachment);
  VkFramebufferCreateInfo framebuffer_info{};
  framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_info.attachmentCount = attachments.size();
  framebuffer_info.pAttachments = attachments.data();
  framebuffer_info.width = extent.width;
  framebuffer_info.height = extent.height;
  framebuffer_info.renderPass = render_pass.Handle();
  framebuffer_info.layers = 1;

  GRASSLAND_VULKAN_CHECK(vkCreateFramebuffer(
      device_.Handle(), &framebuffer_info, nullptr, &framebuffer_));
}

Framebuffer::~Framebuffer() {
  if (framebuffer_ != VK_NULL_HANDLE) {
    vkDestroyFramebuffer(device_.Handle(), framebuffer_, nullptr);
  }
}

}  // namespace grassland::vulkan
