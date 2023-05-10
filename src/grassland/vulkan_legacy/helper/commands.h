#pragma once
#include "grassland/vulkan_legacy/util.h"

namespace grassland::vulkan_legacy::helper {
void CommandBegin(VkCommandBuffer command_buffer);

void CommandEnd(VkCommandBuffer command_buffer);
}  // namespace grassland::vulkan_legacy::helper
