#include "grassland/vulkan/resources/frame_buffer.h"

namespace grassland::vulkan {

Framebuffer::Framebuffer(class grassland::vulkan::Core *core,
                         VkExtent2D extent,
                         VkRenderPass render_pass,
                         const std::vector<VkImageView> &image_views)
    : core_(core), extent_(extent) {
  VkFramebufferCreateInfo framebuffer_create_info{};
  framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebuffer_create_info.renderPass = render_pass;
  framebuffer_create_info.attachmentCount =
      static_cast<uint32_t>(image_views.size());
  framebuffer_create_info.pAttachments = image_views.data();
  framebuffer_create_info.width = extent.width;
  framebuffer_create_info.height = extent.height;
  framebuffer_create_info.layers = 1;

  if (vkCreateFramebuffer(core_->Device()->Handle(), &framebuffer_create_info,
                          nullptr, &framebuffer_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create framebuffer!");
  }
}

Framebuffer::~Framebuffer() {
  vkDestroyFramebuffer(core_->Device()->Handle(), framebuffer_, nullptr);
}

VkFramebuffer Framebuffer::Handle() const {
  return framebuffer_;
}

class Core *Framebuffer::Core() const {
  return core_;
}

VkExtent2D Framebuffer::Extent() const {
  return extent_;
}

}  // namespace grassland::vulkan
