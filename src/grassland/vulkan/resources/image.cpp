#include "grassland/vulkan/resources/image.h"

namespace grassland::vulkan {
Image::Image(Device *device,
             VkImage image,
             VkFormat format,
             VkExtent2D extent,
             VkImageUsageFlags usage,
             VkImageAspectFlags aspect_flags,
             VkSampleCountFlagBits sample_count,
             VkImageLayout layout,
             bool individual_image)
    : device_(device),
      image_(image),
      format_(format),
      extent_(extent),
      usage_(usage),
      aspect_flags_(aspect_flags),
      sample_count_(sample_count),
      layout_(layout),
      individual_image_(individual_image) {
}

Image::~Image() {
  if (individual_image_) {
    vkDestroyImage(device_->Handle(), image_, nullptr);
  }
}

}  // namespace grassland::vulkan
