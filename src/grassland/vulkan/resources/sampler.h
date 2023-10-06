#pragma once

#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class Sampler {
 public:
  Sampler(class Core *core,
          VkFilter mag_filter,
          VkFilter min_filter,
          VkSamplerAddressMode address_mode_u,
          VkSamplerAddressMode address_mode_v,
          VkSamplerAddressMode address_mode_w,
          VkBool32 anisotropy_enable,
          VkBorderColor border_color,
          VkSamplerMipmapMode mipmap_mode);
  ~Sampler();

  [[nodiscard]] VkSampler Handle() const;
  [[nodiscard]] class Core *Core() const;

 private:
  class Core *core_{};
  VkSampler sampler_{};
};
}  // namespace grassland::vulkan
