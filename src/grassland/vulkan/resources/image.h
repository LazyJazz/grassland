#include "grassland/vulkan/core/device.h"

namespace grassland::vulkan {
class Image {
 public:
  Image(Device *device,
        VkImage image,
        VkFormat format,
        VkExtent2D extent,
        VkImageUsageFlags usage,
        VkImageAspectFlags aspect_flags,
        VkSampleCountFlagBits sample_count,
        VkImageLayout layout,
        bool individual_image);
  ~Image();

  [[nodiscard]] VkImage Handle() const {
    return image_;
  }
  [[nodiscard]] VkFormat Format() const {
    return format_;
  }
  [[nodiscard]] VkExtent2D Extent() const {
    return extent_;
  }
  [[nodiscard]] VkImageUsageFlags Usage() const {
    return usage_;
  }
  [[nodiscard]] VkImageAspectFlags AspectFlags() const {
    return aspect_flags_;
  }
  [[nodiscard]] VkSampleCountFlagBits SampleCount() const {
    return sample_count_;
  }
  [[nodiscard]] VkImageLayout Layout() const {
    return layout_;
  }

 private:
  Device *device_;

  VkImage image_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  VkImageUsageFlags usage_{};
  VkImageAspectFlags aspect_flags_{};
  VkSampleCountFlagBits sample_count_{};
  VkImageLayout layout_{};
  bool individual_image_{false};
};
}  // namespace grassland::vulkan
