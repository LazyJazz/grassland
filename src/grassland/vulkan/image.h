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
  VK_HANDLE(VkImage)
  Device *device_{nullptr};
};
}  // namespace grassland::vulkan
