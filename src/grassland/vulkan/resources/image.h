#include "grassland/vulkan/core/device.h"
#include "grassland/vulkan/resources/image_reference.h"

namespace grassland::vulkan {
class Image {
 public:
  Image(Core *core,
        VkFormat format,
        VkExtent2D extent,
        VkImageUsageFlags usage,
        VkImageAspectFlags aspect_flags,
        VkSampleCountFlagBits sample_count,
        VkImageLayout layout);
  ~Image();

  [[nodiscard]] VkImage Handle() const {
    return image_;
  }

  // Complete all the gets
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
  Core *core_{};

  VkImage image_{};
  VkImageView image_view_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  VkImageUsageFlags usage_{};
  VkImageAspectFlags aspect_flags_{};
  VkSampleCountFlagBits sample_count_{};
  VkImageLayout layout_{};

  VmaAllocation allocation_{};
};
}  // namespace grassland::vulkan
