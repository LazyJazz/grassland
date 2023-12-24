#pragma once

#include "grassland/vulkan/resources/buffer.h"
#include "grassland/vulkan/resources/dynamic_object.h"

namespace grassland::vulkan {
template <class Type = uint8_t>
struct DynamicBuffer : public BufferObject, public DynamicObject {
 public:
  DynamicBuffer() = default;

  DynamicBuffer(class Core *core_,
                size_t length = 1,
                VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    Init(core_, length, usage);
  }

  ~DynamicBuffer() {
    DeactiveMap();
  }

  void Init(class Core *core_,
            size_t length = 1,
            VkBufferUsageFlags usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) {
    if (buffers_.empty()) {
      length_ = length;
      buffers_.reserve(core_->MaxFramesInFlight());
      versions_.resize(core_->MaxFramesInFlight(), 0);
      for (size_t i = 0; i < core_->MaxFramesInFlight(); i++) {
        buffers_.emplace_back(std::make_unique<class Buffer>(
            core_, static_cast<VkDeviceSize>(sizeof(Type) * length),
            usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY));
      }
      staging_buffer_ = std::make_unique<class Buffer>(
          core_, static_cast<VkDeviceSize>(sizeof(Type) * length),
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VMA_MEMORY_USAGE_CPU_ONLY);
    }
  }

  bool SyncData(VkCommandBuffer cmd_buffer, uint32_t frame_index) override {
    DeactiveMap();
    if (staging_version_ != versions_[frame_index]) {
      VkBufferCopy copy_region{};
      copy_region.size = sizeof(Type) * length_;
      vkCmdCopyBuffer(cmd_buffer, staging_buffer_->Handle(),
                      buffers_[frame_index]->Handle(), 1, &copy_region);
      versions_[frame_index] = staging_version_;
      return true;
    }
    return false;
  }

  bool SyncData(std::function<void(VkCommandBuffer)> &function,
                uint32_t frame_index) {
    DeactiveMap();
    if (staging_version_ != versions_[frame_index]) {
      function = [&](VkCommandBuffer cmd_buffer) {
        VkBufferCopy copy_region{};
        copy_region.size = sizeof(Type) * length_;
        vkCmdCopyBuffer(cmd_buffer, staging_buffer_->Handle(),
                        buffers_[frame_index]->Handle(), 1, &copy_region);
        versions_[frame_index] = staging_version_;
      };
      return true;
    }
    return false;
  }

  void UploadContents(const Type *contents,
                      size_t length = 1,
                      size_t offset = 0) {
    UploadData(reinterpret_cast<const uint8_t *>(contents),
               sizeof(Type) * length, sizeof(Type) * offset);
  }

  void UploadData(const uint8_t *data, size_t size, size_t offset = 0) {
    ActiveMap();
    memcpy(mapped_data_ + offset, data, size);
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

  size_t Length() const {
    return length_;
  }

  VkDeviceSize Size() const {
    return staging_buffer_->Size();
  }

  Buffer *GetBuffer(int frame_index) const override {
    return buffers_[frame_index].get();
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
      staging_version_++;
    }
  }

  std::vector<std::unique_ptr<class Buffer>> buffers_;
  std::vector<uint64_t> versions_;
  std::unique_ptr<class Buffer> staging_buffer_;

  Type *mapped_data_{nullptr};
  size_t length_{};
  uint64_t staging_version_{0};
};
}  // namespace grassland::vulkan
