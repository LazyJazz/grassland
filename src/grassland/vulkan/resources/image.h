#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class Image {
 public:
  Image(class Core *core,
        VkFormat format,
        VkExtent2D extent,
        VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_STORAGE_BIT,
        VkImageAspectFlags aspect_flags = VK_IMAGE_ASPECT_COLOR_BIT,
        VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT);
  ~Image();

  [[nodiscard]] class Core *Core() const {
    return core_;
  }

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

 private:
  class Core *core_{};

  VkImage image_{};
  VkImageView image_view_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  VkImageUsageFlags usage_{};
  VkImageAspectFlags aspect_flags_{};
  VkSampleCountFlagBits sample_count_{};

  VmaAllocation allocation_{};
};

void TransitImageLayout(VkCommandBuffer command_buffer,
                        VkImage image,
                        VkImageLayout old_layout,
                        VkImageLayout new_layout,
                        VkPipelineStageFlags src_stage_flags,
                        VkPipelineStageFlags dst_stage_flags,
                        VkAccessFlags src_access_flags,
                        VkAccessFlags dst_access_flags,
                        VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT);

}  // namespace grassland::vulkan
