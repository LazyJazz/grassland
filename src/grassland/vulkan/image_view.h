#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class ImageView {
 public:
  ImageView(Device *device, VkImage image, VkFormat format);
  ~ImageView();

 private:
  VK_HANDLE(VkImageView)
  Device *device_{nullptr};
};
}  // namespace grassland::vulkan
