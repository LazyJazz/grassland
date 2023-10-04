#pragma once
#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy::helper {
void CommandBegin(VkCommandBuffer command_buffer);

void CommandEnd(VkCommandBuffer command_buffer);
}  // namespace grassland::vulkan_legacy::helper
