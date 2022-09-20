#pragma once
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class ImageView {
 public:
  ImageView();
  explicit ImageView(const VkImageView &image_view);
  ~ImageView();

 private:
  VK_HANDLE(VkImageView)
};
}  // namespace grassland::vulkan
