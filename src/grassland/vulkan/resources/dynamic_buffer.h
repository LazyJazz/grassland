#pragma once

#include <grassland/vulkan/resources/buffer.h>

namespace grassland::vulkan {
template <typename T>
class DynamicBuffer {
 public:
  explicit DynamicBuffer(
      class Core *core,
      size_t size = 0,
      VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                 VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
      : core_(core), size_(size), usage_(usage) {
    for (int i = 0; i < core_->MaxFramesInFlight(); i++) {
      buffers_.emplace_back(std::make_unique<Buffer>(
          core, sizeof(T), usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VMA_MEMORY_USAGE_GPU_ONLY));
    }
    capacity_ = std::max(size, size_t(1));
    host_buffer_ = std::make_unique<Buffer>(core_, sizeof(T) * capacity_,
                                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                            VMA_MEMORY_USAGE_CPU_TO_GPU);
  }

  T *Data() {
    AquireMapping();
    return mapped_data_;
  }

  const T *Data() const {
    AquireMapping();
    return mapped_data_;
  }

  T &operator[](size_t index) {
    AquireMapping();
    return mapped_data_[index];
  }

  void PushBack(const T &value) {
    if (size_ == capacity_) {
      ExtendCapacity();
    }
    AquireMapping();
    mapped_data_[size_++] = value;
  }

  void PopBack() {
    if (size_ > 0) {
      size_--;
    }
  }

  void Resize(size_t size);
  void Reserve(size_t capacity) {
    if (capacity > capacity_) {
      ReallocateCapacity(capacity);
    }
  }

  void Clear() {
    size_ = 0;
  }

  void CopyFrom(const std::vector<T> &data) {
    if (data.size() > capacity_) {
      ReallocateCapacity(data.size());
    }
    size_ = data.size();
    AquireMapping();
    memcpy(mapped_data_, data.data(), sizeof(T) * size_);
  }

  void CopyFrom(const T *data, size_t size) {
    if (size > capacity_) {
      ReallocateCapacity(size);
    }
    size_ = size;
    AquireMapping();
    memcpy(mapped_data_, data, sizeof(T) * size_);
  }

  void CopyFrom(const T *data, size_t size, size_t offset) {
    if (size + offset > capacity_) {
      ReallocateCapacity(size + offset);
    }
    size_ = size + offset;
    AquireMapping();
    memcpy(mapped_data_ + offset, data, sizeof(T) * size);
  }

  void CopyFrom(const std::vector<T> &data, size_t offset) {
    if (data.size() + offset > capacity_) {
      ReallocateCapacity(data.size() + offset);
    }
    size_ = data.size() + offset;
    AquireMapping();
    memcpy(mapped_data_ + offset, data.data(), sizeof(T) * data.size());
  }

  void CopyFrom(const DynamicBuffer<T> &buffer) {
    if (buffer.size_ > capacity_) {
      ReallocateCapacity(buffer.size_);
    }
    size_ = buffer.size_;
    AquireMapping();
    memcpy(mapped_data_, buffer.Data(), sizeof(T) * size_);
  }

  void CopyFrom(const DynamicBuffer<T> &buffer, size_t offset) {
    if (buffer.size_ + offset > capacity_) {
      ReallocateCapacity(buffer.size_ + offset);
    }
    size_ = buffer.size_ + offset;
    AquireMapping();
    memcpy(mapped_data_ + offset, buffer.Data(), sizeof(T) * buffer.size_);
  }

  void PreRender() {
    auto current_frame = core_->CurrentFrame();
    auto &buffer = buffers_[current_frame];
    // Copy data from mapped_data_ to buffer, if buffer size is not enough,
    // extend it first
    if (size_ * sizeof(T) > buffer->Size()) {
      buffer = std::make_unique<Buffer>(
          core_, sizeof(T) * size_, usage_ | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          VMA_MEMORY_USAGE_GPU_ONLY);
    }
    core_->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
      VkBufferCopy copy_region{};
      copy_region.size = sizeof(T) * size_;
      vkCmdCopyBuffer(cmd_buffer, host_buffer_->Handle(), buffer->Handle(), 1,
                      &copy_region);
    });
  }

 private:
  void AquireMapping();
  void ReleaseMapping();
  void ReallocateCapacity(size_t new_capacity);
  void ExtendCapacity();

  class Core *core_{};
  size_t size_{};
  size_t capacity_{};
  VkBufferUsageFlags usage_{};
  std::vector<std::unique_ptr<Buffer>> buffers_{};
  std::unique_ptr<Buffer> host_buffer_{};
  T *mapped_data_{};
};

template <typename T>
void DynamicBuffer<T>::Resize(size_t size) {
  if (size > capacity_) {
    ReallocateCapacity(size);
  }
  size_ = size;
}

template <typename T>
void DynamicBuffer<T>::ReallocateCapacity(size_t new_capacity) {
  ReleaseMapping();
  auto new_buffer_ = std::make_unique<Buffer>(core_, sizeof(T) * new_capacity,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                              VMA_MEMORY_USAGE_CPU_TO_GPU);
  // Copy buffer from old to new
  core_->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
    VkBufferCopy copy_region{};
    copy_region.size = sizeof(T) * std::min(size_, new_capacity);
    vkCmdCopyBuffer(cmd_buffer, host_buffer_->Handle(), new_buffer_->Handle(),
                    1, &copy_region);
  });

  host_buffer_ = std::move(new_buffer_);
  capacity_ = new_capacity;
}

template <typename T>
void DynamicBuffer<T>::ExtendCapacity() {
  ReallocateCapacity(capacity_ * 2);
}

template <typename T>
void DynamicBuffer<T>::AquireMapping() {
  if (mapped_data_ == nullptr) {
    mapped_data_ = reinterpret_cast<T *>(host_buffer_->Map());
  }
}

template <typename T>
void DynamicBuffer<T>::ReleaseMapping() {
  if (mapped_data_ != nullptr) {
    host_buffer_->Unmap();
    mapped_data_ = nullptr;
  }
}
}  // namespace grassland::vulkan
