#pragma once
#include "device.h"
#include "device_memory.h"

namespace grassland::vulkan {
class Image {
 public:
  GRASSLAND_CANNOT_COPY(Image);
  Image(const class Device &device, VkExtent2D extent, VkFormat format);

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkImage, image_)
};

}  // namespace grassland::vulkan
