#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>
namespace grassland::vulkan {
class Image {
 public:
  Image();
  explicit Image(const VkImage &image);
  ~Image();

 private:
  GRASSLAND_VULKAN_HANDLE(VkImage)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
