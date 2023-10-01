#pragma once
#include <grassland/vulkan_legacy/physical_device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy::helper {

VkFormat FindSupportedFormat(PhysicalDevice *physical_device,
                             const std::vector<VkFormat> &candidates,
                             VkImageTiling tiling,
                             VkFormatFeatureFlags features);

VkFormat FindDepthFormat(PhysicalDevice *physical_device);
}  // namespace grassland::vulkan_legacy::helper
