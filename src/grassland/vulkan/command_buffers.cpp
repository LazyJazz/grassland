#include "command_buffers.h"

namespace grassland::vulkan {

CommandBuffers::CommandBuffers(const CommandPool &command_pool, uint32_t size)
    : command_pool_(command_pool) {
  VkCommandBufferAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.commandPool = command_pool_.Handle();
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandBufferCount = size;

  command_buffers_.resize(size);

  GRASSLAND_VULKAN_CHECK(vkAllocateCommandBuffers(
      command_pool_.Device().Handle(), &alloc_info, command_buffers_.data()));
}

CommandBuffers::~CommandBuffers() {
  if (!command_buffers_.empty()) {
    vkFreeCommandBuffers(command_pool_.Device().Handle(),
                         command_pool_.Handle(), command_buffers_.size(),
                         command_buffers_.data());
    command_buffers_.clear();
  }
}

VkCommandBuffer CommandBuffers::Begin(size_t i) {
  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  GRASSLAND_VULKAN_CHECK(
      vkBeginCommandBuffer(command_buffers_[i], &begin_info));

  return command_buffers_[i];
}

void CommandBuffers::End(size_t i) {
  GRASSLAND_VULKAN_CHECK(vkEndCommandBuffer(command_buffers_[i]));
}

}  // namespace grassland::vulkan
