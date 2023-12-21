#pragma once

#include "grassland/vulkan/resources/buffer.h"

namespace grassland::vulkan {
template <class Type = uint8_t>
struct StaticBuffer {
 public:
  StaticBuffer(class Core *core_,
               size_t length = 1,
               VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    buffer_ = std::make_unique<class Buffer>(
        core_, static_cast<VkDeviceSize>(sizeof(Type) * length), usage,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
  }

  void UploadContents(const Type *contents,
                      size_t length,
                      size_t offset = 0) const {
    UploadData(reinterpret_cast<const uint8_t *>(contents),
               sizeof(Type) * length, sizeof(Type) * offset);
  }

  void UploadData(const uint8_t *data, size_t size, size_t offset = 0) const {
    // Upload with staging buffer
    class Buffer staging_buffer(
        buffer_->Core(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
    memcpy(staging_buffer.Map(), data, size);
    staging_buffer.Unmap();
    buffer_->Core()->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
      VkBufferCopy copy_region{};
      copy_region.size = size;
      copy_region.dstOffset = offset;
      vkCmdCopyBuffer(cmd_buffer, staging_buffer.Handle(), buffer_->Handle(), 1,
                      &copy_region);
    });
  }

  void DownloadContents(Type *contents,
                        size_t length,
                        size_t offset = 0) const {
    DownloadData(reinterpret_cast<uint8_t *>(contents), sizeof(Type) * length,
                 sizeof(Type) * offset);
  }

  void DownloadData(uint8_t *data, size_t size, size_t offset = 0) const {
    // Download with staging buffer
    class Buffer staging_buffer(
        buffer_->Core(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO,
        VMA_ALLOCATION_CREATE_MAPPED_BIT |
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    buffer_->Core()->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
      VkBufferCopy copy_region{};
      copy_region.size = size;
      copy_region.srcOffset = offset;
      vkCmdCopyBuffer(cmd_buffer, buffer_->Handle(), staging_buffer.Handle(), 1,
                      &copy_region);
    });
    memcpy(data, staging_buffer.Map(), size);
    staging_buffer.Unmap();
  }

  [[nodiscard]] class Buffer *Buffer() const {
    return buffer_.get();
  }

 private:
  std::unique_ptr<class Buffer> buffer_{};
};
}  // namespace grassland::vulkan
