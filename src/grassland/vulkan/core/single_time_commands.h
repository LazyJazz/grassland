#pragma once
#include "grassland/vulkan/core/command_buffer.h"

namespace grassland::vulkan {
// Define a function to execute a single command
void SingleTimeCommands(CommandBuffer *command_buffer,
                        const std::function<void(VkCommandBuffer)> &function);

void SingleTimeCommands(CommandPool *command_pool,
                        const std::function<void(VkCommandBuffer)> &function);

void SingleTimeCommands(Device *device,
                        const std::function<void(VkCommandBuffer)> &function);
}  // namespace grassland::vulkan
