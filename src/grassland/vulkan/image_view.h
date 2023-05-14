#pragma once

#include "device.h"
#include "image.h"

namespace grassland::vulkan {
class ImageView {
 public:
  GRASSLAND_CANNOT_COPY(ImageView)
  explicit ImageView(const class Device &device,
                     VkImage image,
                     VkFormat format,
                     VkImageAspectFlags aspect_flags);
  ~ImageView();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkImageView, image_view_)
};
}  // namespace grassland::vulkan
