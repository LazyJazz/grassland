#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>
namespace grassland::vulkan {
class Image {
 public:
  Image();
  ~Image();

 private:
  GRASSLAND_VULKAN_HANDLE(VkImage)
  GRASSLAND_VULKAN_DEVICE_PTR
  VkDeviceMemory device_memory_{};
  uint32_t width_{};
  uint32_t height_{};
  VkFormat format_{};
};
}  // namespace grassland::vulkan
