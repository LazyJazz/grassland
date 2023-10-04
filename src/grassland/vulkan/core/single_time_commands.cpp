#include "grassland/vulkan/core/single_time_commands.h"

namespace grassland::vulkan {
void SingleTimeCommands(CommandBuffer *command_buffer,
                        const std::function<void(VkCommandBuffer)> &function) {
  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer->Handle(), &begin_info);

  function(command_buffer->Handle());

  vkEndCommandBuffer(command_buffer->Handle());

  VkCommandBuffer command_buffers[] = {command_buffer->Handle()};
  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = command_buffers;

  const auto single_time_command_queue =
      command_buffer->CommandPool()->Device()->SingleTimeCommandQueue();

  vkQueueSubmit(single_time_command_queue.Handle(), 1, &submit_info, nullptr);
  single_time_command_queue.WaitIdle();
}

void SingleTimeCommands(CommandPool *command_pool,
                        const std::function<void(VkCommandBuffer)> &function) {
  // Create a command buffer and call the function above
  std::unique_ptr<CommandBuffer> command_buffer =
      std::make_unique<CommandBuffer>(command_pool);
  SingleTimeCommands(command_buffer.get(), function);
}

void SingleTimeCommands(Device *device,
                        const std::function<void(VkCommandBuffer)> &function) {
  // Create a command pool and call the function above
  std::unique_ptr<CommandPool> command_pool =
      std::make_unique<CommandPool>(device);
  SingleTimeCommands(command_pool.get(), function);
}

}  // namespace grassland::vulkan
