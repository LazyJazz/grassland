#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class ImageView {
 public:
  ImageView(Device *device, VkImage image, VkFormat format);
  ~ImageView();

 private:
  GRASSLAND_VULKAN_HANDLE(VkImageView)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
