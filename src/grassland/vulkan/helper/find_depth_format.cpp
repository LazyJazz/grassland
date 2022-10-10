#include <grassland/util/logging.h>
#include <grassland/vulkan/helper/find_depth_format.h>

namespace grassland::vulkan::helper {

VkFormat FindSupportedFormat(PhysicalDevice *physical_device,
                             const std::vector<VkFormat> &candidates,
                             VkImageTiling tiling,
                             VkFormatFeatureFlags features) {
  {
    for (VkFormat format : candidates) {
      VkFormatProperties props;
      vkGetPhysicalDeviceFormatProperties(physical_device->GetHandle(), format,
                                          &props);

      if (tiling == VK_IMAGE_TILING_LINEAR &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                 (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    LAND_ERROR("[Vulkan] failed to find supported format!");
  }
}

VkFormat FindDepthFormat(PhysicalDevice *physical_device) {
  return FindSupportedFormat(
      physical_device,
      {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT,
       VK_FORMAT_D24_UNORM_S8_UINT},
      VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

}  // namespace grassland::vulkan::helper
