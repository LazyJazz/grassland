#pragma once
#include <grassland/vulkan_legacy/device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class Sampler {
 public:
  explicit Sampler(
      Device *device,
      VkFilter filter = VK_FILTER_LINEAR,
      VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VkBool32 anisotropy_enable = VK_TRUE,
      VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
      VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR);
  Sampler(Device *device,
          VkFilter mag_filter,
          VkFilter min_filter,
          VkSamplerAddressMode address_mode_u,
          VkSamplerAddressMode address_mode_v,
          VkSamplerAddressMode address_mode_w,
          VkBool32 anisotropy_enable,
          VkBorderColor border_color,
          VkSamplerMipmapMode mipmap_mode);
  ~Sampler();
  void Reset(
      VkFilter filter = VK_FILTER_LINEAR,
      VkSamplerAddressMode address_mode = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      VkBool32 anisotropy_enable = VK_TRUE,
      VkBorderColor border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
      VkSamplerMipmapMode mipmap_mode = VK_SAMPLER_MIPMAP_MODE_LINEAR);
  void Reset(VkFilter mag_filter,
             VkFilter min_filter,
             VkSamplerAddressMode address_mode_u,
             VkSamplerAddressMode address_mode_v,
             VkSamplerAddressMode address_mode_w,
             VkBool32 anisotropy_enable,
             VkBorderColor border_color,
             VkSamplerMipmapMode mipmap_mode);

 private:
  void Clear();
  GRASSLAND_VULKAN_HANDLE(VkSampler)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
