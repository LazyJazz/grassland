#include "grassland/vulkan/core/command_buffer.h"

namespace grassland::vulkan {
CommandBuffer::CommandBuffer(class CommandPool *command_pool)
    : command_pool_(command_pool) {
  VkCommandBufferAllocateInfo allocate_info{};
  allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocate_info.commandPool = command_pool_->Handle();
  allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocate_info.commandBufferCount = 1;
  vkAllocateCommandBuffers(command_pool_->Device()->Handle(), &allocate_info,
                           &command_buffer_);
}
CommandBuffer::~CommandBuffer() {
  vkFreeCommandBuffers(command_pool_->Device()->Handle(),
                       command_pool_->Handle(), 1, &command_buffer_);
}
VkCommandBuffer CommandBuffer::Handle() const {
  return command_buffer_;
}
class CommandPool *CommandBuffer::CommandPool() const {
  return command_pool_;
}

}  // namespace grassland::vulkan
