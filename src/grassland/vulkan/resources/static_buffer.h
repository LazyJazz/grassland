#pragma once
#include "grassland/vulkan/resources/buffer.h"

namespace grassland::vulkan {
template <typename T>
class StaticBuffer {
 public:
  StaticBuffer(class Core *core,
               size_t size,
               VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
      : core_(core),
        size_(size),
        buffer_(core, size * sizeof(T), usage, VMA_MEMORY_USAGE_GPU_ONLY) {
  }
  [[nodiscard]] class Core *Core() const {
    return core_;
  }
  [[nodiscard]] VkBuffer Handle() {
    return buffer_.Handle();
  }
  [[nodiscard]] VmaAllocation Allocation() {
    return buffer_.Allocation();
  }
  [[nodiscard]] size_t Size() {
    return size_;
  }
  [[nodiscard]] VkDeviceSize BufferSize() {
    return buffer_.Size();
  }

 private:
  class Core *core_{};
  size_t size_{};
  class Buffer buffer_;
};
}  // namespace grassland::vulkan
