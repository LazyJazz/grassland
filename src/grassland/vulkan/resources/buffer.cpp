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

void CopyBuffer(VkCommandBuffer command_buffer,
                Buffer *src_buffer,
                Buffer *dst_buffer,
                VkDeviceSize size) {
  VkBufferCopy copy_region{};
  copy_region.size = size;
  vkCmdCopyBuffer(command_buffer, src_buffer->Handle(), dst_buffer->Handle(), 1,
                  &copy_region);
}

void UploadBuffer(Buffer *buffer, const void *data, VkDeviceSize size) {
  Buffer staging_buffer(buffer->Core(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_CPU_ONLY);
  void *staging_data = staging_buffer.Map();
  memcpy(staging_data, data, size);
  staging_buffer.Unmap();

  buffer->Core()->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
    CopyBuffer(cmd_buffer, &staging_buffer, buffer, size);
  });
}

void DownloadBuffer(Buffer *buffer, void *data, VkDeviceSize size) {
  Buffer staging_buffer(buffer->Core(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VMA_MEMORY_USAGE_CPU_ONLY);
  buffer->Core()->SingleTimeCommands([&](VkCommandBuffer cmd_buffer) {
    CopyBuffer(cmd_buffer, buffer, &staging_buffer, size);
  });
  void *staging_data = staging_buffer.Map();
  memcpy(data, staging_data, size);
  staging_buffer.Unmap();
}

}  // namespace grassland::vulkan
