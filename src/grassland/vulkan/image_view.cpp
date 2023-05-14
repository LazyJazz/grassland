#include "image_view.h"

namespace grassland::vulkan {

ImageView::ImageView(const class Device &device,
                     VkImage image,
                     VkFormat format,
                     VkImageAspectFlags aspect_flags)
    : device_(device) {
  VkImageViewCreateInfo view_info{};
  view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image = image;
  view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format = format;
  view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  view_info.subresourceRange.aspectMask = aspect_flags;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.baseMipLevel = 0;
  view_info.subresourceRange.layerCount = 1;
  view_info.subresourceRange.levelCount = 1;

  GRASSLAND_VULKAN_CHECK(
      vkCreateImageView(device_.Handle(), &view_info, nullptr, &image_view_));
}

ImageView::~ImageView() {
  if (image_view_ != VK_NULL_HANDLE) {
    vkDestroyImageView(device_.Handle(), image_view_, nullptr);
  }
}

}  // namespace grassland::vulkan
