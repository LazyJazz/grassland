#include "sampler.h"

namespace grassland::vulkan {

Sampler::Sampler(const class grassland::vulkan::Device &device,
                 VkFilter filter,
                 VkSamplerAddressMode address_mode,
                 VkBool32 anisotropy_enable,
                 VkBorderColor border_color,
                 VkSamplerMipmapMode mipmap_mode)
    : Sampler(device,
              filter,
              filter,
              address_mode,
              address_mode,
              address_mode,
              anisotropy_enable,
              border_color,
              mipmap_mode) {
}

Sampler::Sampler(const class grassland::vulkan::Device &device,
                 VkFilter mag_filter,
                 VkFilter min_filter,
                 VkSamplerAddressMode address_mode_u,
                 VkSamplerAddressMode address_mode_v,
                 VkSamplerAddressMode address_mode_w,
                 VkBool32 anisotropy_enable,
                 VkBorderColor border_color,
                 VkSamplerMipmapMode mipmap_mode)
    : device_(device) {
  if (!device_.PhysicalDevice().GetFeatures().samplerAnisotropy) {
    anisotropy_enable = VK_FALSE;
  }
  VkSamplerCreateInfo sampler_info{};
  sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_info.magFilter = mag_filter;
  sampler_info.minFilter = min_filter;
  sampler_info.addressModeU = address_mode_u;
  sampler_info.addressModeV = address_mode_v;
  sampler_info.addressModeW = address_mode_w;
  sampler_info.anisotropyEnable = anisotropy_enable;
  sampler_info.maxAnisotropy =
      anisotropy_enable
          ? device_.PhysicalDevice().GetProperties().limits.maxSamplerAnisotropy
          : 1.0f;
  sampler_info.borderColor = border_color;
  sampler_info.unnormalizedCoordinates = VK_FALSE;
  sampler_info.compareEnable = VK_FALSE;
  sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_info.mipmapMode = mipmap_mode;

  GRASSLAND_VULKAN_CHECK(
      vkCreateSampler(device_.Handle(), &sampler_info, nullptr, &sampler_));
}

}  // namespace grassland::vulkan
