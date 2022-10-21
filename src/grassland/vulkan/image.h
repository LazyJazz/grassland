#pragma once
#include <grassland/vulkan/buffer.h>
#include <grassland/vulkan/command_buffer.h>
#include <grassland/vulkan/command_pool.h>
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/queue.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Image {
 public:
  /*
   * Notice: SRGB formats cannot work with VK_IMAGE_USAGE_STORAGE_BIT, use UNORM
   * format instead.
   * */
  Image(Device *device,
        uint32_t width,
        uint32_t height,
        VkFormat format = VK_FORMAT_R8G8B8A8_SRGB,
        VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL);
  ~Image();
  [[nodiscard]] VkFormat GetFormat() const;
  [[nodiscard]] uint32_t GetWidth() const;
  [[nodiscard]] uint32_t GetHeight() const;
  void TransitImageLayout(CommandBuffer *command_buffer,
                          VkImageLayout new_layout,
                          VkPipelineStageFlags new_stage_flags,
                          VkAccessFlags new_access_flags);
  void Update(CommandBuffer *command_buffer, Buffer *buffer);
  void Retrieve(CommandBuffer *command_buffer, Buffer *buffer);
  VkImageLayout GetImageLayout();

 private:
  GRASSLAND_VULKAN_HANDLE(VkImage)
  GRASSLAND_VULKAN_DEVICE_PTR
  VkDeviceMemory device_memory_{};
  uint32_t width_{};
  uint32_t height_{};
  VkFormat format_{VK_FORMAT_R32G32B32A32_SFLOAT};
  VkImageLayout image_layout_{VK_IMAGE_LAYOUT_UNDEFINED};
  VkPipelineStageFlags pipeline_stage_flags_{VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT};
  VkAccessFlags access_flags_{0};
};

void UploadImage(Queue *graphics_queue,
                 CommandPool *command_pool,
                 Image *image,
                 Buffer *buffer);

void DownloadImage(Queue *graphics_queue,
                   CommandPool *command_pool,
                   Image *image,
                   Buffer *buffer);

void UploadImage(CommandPool *command_pool, Image *image, Buffer *buffer);

void DownloadImage(CommandPool *command_pool, Image *image, Buffer *buffer);
}  // namespace grassland::vulkan
