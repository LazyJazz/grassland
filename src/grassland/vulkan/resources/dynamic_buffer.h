#pragma once

#include "grassland/vulkan/resources/buffer.h"

namespace grassland::vulkan {
template <class Type = uint8_t>
struct DynamicBuffer {
 public:
  DynamicBuffer(class Core *core_,
                size_t length = 1,
                VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    buffers_.reserve(length);
    for (size_t i = 0; i < core_->MaxFramesInFlight(); i++) {
      buffers_.emplace_back(std::make_unique<class Buffer>(
          core_, static_cast<VkDeviceSize>(sizeof(Type) * length), usage,
          VMA_MEMORY_USAGE_GPU_ONLY));
    }
    staging_buffer_ = std::make_unique<class Buffer>(
        core_, static_cast<VkDeviceSize>(sizeof(Type) * length),
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
  }

  Type &operator[](size_t index) {
    return At(index);
  }

  Type &At(size_t index) {
    ActiveMap();
    return mapped_data_[index];
  }

  Type *Data() {
    ActiveMap();
    return mapped_data_;
  }

  class Buffer *Buffer(size_t index) {
    return buffers_[index].get();
  }

 private:
  void ActiveMap() {
    if (!mapped_data_) {
      mapped_data_ = reinterpret_cast<Type *>(staging_buffer_->Map());
    }
  }

  void DeactiveMap() {
    if (mapped_data_) {
      staging_buffer_->Unmap();
      mapped_data_ = nullptr;
    }
  }

  std::vector<std::unique_ptr<class Buffer>> buffers_;
  std::unique_ptr<class Buffer> staging_buffer_;

  Type *mapped_data_{nullptr};
};
}  // namespace grassland::vulkan
