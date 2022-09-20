#pragma once
#include <grassland/vulkan/util.h>
namespace grassland::vulkan {
class Image {
 public:
  Image();
  explicit Image(const VkImage &image);
  ~Image();

 private:
  VK_HANDLE(VkImage)
};
}  // namespace grassland::vulkan
