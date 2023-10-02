#include "image_reference.h"

namespace grassland::vulkan {
ImageReference::ImageReference(Device *device,
                               VkImage image,
                               VkImageView image_view,
                               VkFormat format,
                               VkExtent2D extent,
                               VkImageUsageFlags usage,
                               VkImageAspectFlags aspect_flags,
                               VkSampleCountFlagBits sample_count,
                               VkImageLayout layout)
    : device_(device),
      image_(image),
      image_view_(image_view),
      format_(format),
      extent_(extent),
      usage_(usage),
      aspect_flags_(aspect_flags),
      sample_count_(sample_count),
      layout_(layout) {
}

ImageReference::~ImageReference() = default;

}  // namespace grassland::vulkan
