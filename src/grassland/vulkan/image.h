#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>
#include <stb_image.h>
#include <stb_image_write.h>
namespace grassland::vulkan {
class Image {
 public:
  Image(Device *device,
        uint32_t width,
        uint32_t height,
        VkFormat format = VK_FORMAT_R8G8B8A8_SRGB,
        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_STORAGE_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL);
  ~Image();
  [[nodiscard]] VkFormat GetFormat() const;
  [[nodiscard]] uint32_t GetWidth() const;
  [[nodiscard]] uint32_t GetHeight() const;

 private:
  GRASSLAND_VULKAN_HANDLE(VkImage)
  GRASSLAND_VULKAN_DEVICE_PTR
  VkDeviceMemory device_memory_{};
  uint32_t width_{};
  uint32_t height_{};
  VkFormat format_{VK_FORMAT_R32G32B32A32_SFLOAT};
};
}  // namespace grassland::vulkan
