#pragma once
#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class ImageReference {
 public:
  ImageReference(Device *device,
                 VkImage image,
                 VkImageView image_view,
                 VkFormat format,
                 VkExtent2D extent,
                 VkImageUsageFlags usage,
                 VkImageAspectFlags aspect_flags,
                 VkSampleCountFlagBits sample_count,
                 VkImageLayout layout);
  ~ImageReference();

  [[nodiscard]] VkImage Image() const {
    return image_;
  }
  [[nodiscard]] VkImageView ImageView() const {
    return image_view_;
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
  VkImageView image_view_;
  VkFormat format_{};
  VkExtent2D extent_{};
  VkImageUsageFlags usage_{};
  VkImageAspectFlags aspect_flags_{};
  VkSampleCountFlagBits sample_count_{};
  VkImageLayout layout_{};
};
}  // namespace grassland::vulkan
