#include "grassland/vulkan/resources/image.h"

namespace grassland::vulkan {

Image::Image(Core *core,
             VkFormat format,
             VkExtent2D extent,
             VkImageUsageFlags usage,
             VkImageAspectFlags aspect_flags,
             VkSampleCountFlagBits sample_count,
             VkImageLayout layout)
    : core_(core),
      format_(format),
      extent_(extent),
      usage_(usage),
      aspect_flags_(aspect_flags),
      sample_count_(sample_count),
      layout_(layout) {
  // Create an image with given parameters
  VkImageCreateInfo image_create_info{};
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = format_;
  image_create_info.extent.width = extent_.width;
  image_create_info.extent.height = extent_.height;
  image_create_info.extent.depth = 1;
  image_create_info.mipLevels = 1;
  image_create_info.arrayLayers = 1;
  image_create_info.samples = sample_count_;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = usage_;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  // Create image from image create info by VMA library

  VmaAllocationCreateInfo allocationInfo = {};
  allocationInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;

  vmaCreateImage(core_->Device()->Allocator(), &image_create_info,
                 &allocationInfo, &image_, &allocation_, nullptr);

  // Create Image View
  VkImageViewCreateInfo image_view_create_info{};
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.image = image_;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = format_;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange.aspectMask = aspect_flags_;
  image_view_create_info.subresourceRange.baseMipLevel = 0;
  image_view_create_info.subresourceRange.levelCount = 1;
  image_view_create_info.subresourceRange.baseArrayLayer = 0;
  image_view_create_info.subresourceRange.layerCount = 1;

  if (vkCreateImageView(core_->Device()->Handle(), &image_view_create_info,
                        nullptr, &image_view_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create image view!");
  }
}

Image::~Image() {
  vkDestroyImageView(core_->Device()->Handle(), image_view_, nullptr);
  vmaDestroyImage(core_->Device()->Allocator(), image_, allocation_);
}

}  // namespace grassland::vulkan
