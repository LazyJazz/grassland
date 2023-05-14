#pragma once
#include "device.h"
#include "device_memory.h"

namespace grassland::vulkan {
class Image {
 public:
  GRASSLAND_CANNOT_COPY(Image);
  Image(const class Device &device,
        VkExtent2D extent,
        VkFormat format,
        VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL,
        VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_STORAGE_BIT |
                                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  ~Image();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkImage, image_)
};

}  // namespace grassland::vulkan
