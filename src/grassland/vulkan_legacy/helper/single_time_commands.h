#pragma once
#include <grassland/vulkan_legacy/command_buffer.h>
#include <grassland/vulkan_legacy/command_pool.h>

namespace grassland::vulkan_legacy::helper {
void SingleTimeCommands(
    CommandPool *command_pool,
    std::function<void(VkCommandBuffer command_buffer)> actions);
void SingleTimeCommands(
    CommandPool *command_pool,
    std::function<void(CommandBuffer *command_buffer)> actions);
}  // namespace grassland::vulkan_legacy::helper
