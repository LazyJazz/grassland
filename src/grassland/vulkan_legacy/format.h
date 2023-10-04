#pragma once
#include <grassland/vulkan_legacy/vulkan.h>

namespace grassland::vulkan_legacy {
uint32_t FormatSizeInBytes(VkFormat format);
uint32_t FormatSlot(VkFormat format);
}  // namespace grassland::vulkan_legacy
