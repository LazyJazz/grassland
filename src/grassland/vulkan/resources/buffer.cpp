#include "grassland/vulkan/resources/buffer.h"

namespace grassland::vulkan {
Buffer::Buffer(struct Core *core,
               VkDeviceSize size,
               VkBufferUsageFlags usage,
               VmaMemoryUsage memory_usage)
    : core_(core), size_(size) {
  // Create buffer with VMA
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = usage;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo alloc_info{};
  alloc_info.usage = memory_usage;

  if (vmaCreateBuffer(core_->Device()->Allocator(), &buffer_info, &alloc_info,
                      &buffer_, &allocation_, nullptr) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create buffer!");
  }
}

Buffer::~Buffer() {
  vmaDestroyBuffer(core_->Device()->Allocator(), buffer_, allocation_);
}

class Core *Buffer::Core() const {
  return core_;
}
VkBuffer Buffer::Handle() const {
  return buffer_;
}

VmaAllocation Buffer::Allocation() const {
  return allocation_;
}

VkDeviceSize Buffer::Size() const {
  return size_;
}

void *Buffer::Map() const {
  void *data;
  vmaMapMemory(core_->Device()->Allocator(), allocation_, &data);
  return data;
}

void Buffer::Unmap() const {
  vmaUnmapMemory(core_->Device()->Allocator(), allocation_);
}
}  // namespace grassland::vulkan
