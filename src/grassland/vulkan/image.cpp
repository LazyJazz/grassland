#include "image.h"

namespace grassland::vulkan {
Image::Image(const class Device &device,
             VkExtent2D extent,
             VkFormat format,
             VkImageTiling tiling,
             VkImageUsageFlags usage)
    : device_(device) {
  VkImageCreateInfo image_info{};
  image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_info.format = format;
  image_info.extent.width = extent.width;
  image_info.extent.height = extent.height;
  image_info.extent.depth = 1;
  image_info.arrayLayers = 1;
  image_info.mipLevels = 1;
  image_info.imageType = VK_IMAGE_TYPE_2D;
  image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_info.samples = VK_SAMPLE_COUNT_1_BIT;
  GRASSLAND_VULKAN_CHECK(
      vkCreateImage(device_.Handle(), &image_info, nullptr, &image_));
}

Image::~Image() {
  if (image_ != VK_NULL_HANDLE) {
    vkDestroyImage(device_.Handle(), image_, nullptr);
  }
}
}  // namespace grassland::vulkan
