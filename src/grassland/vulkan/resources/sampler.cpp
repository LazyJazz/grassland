#include "grassland/vulkan/resources/sampler.h"

namespace grassland::vulkan {
Sampler::Sampler(class Core *core,
                 VkFilter mag_filter,
                 VkFilter min_filter,
                 VkSamplerAddressMode address_mode_u,
                 VkSamplerAddressMode address_mode_v,
                 VkSamplerAddressMode address_mode_w,
                 VkBool32 anisotropy_enable,
                 VkBorderColor border_color,
                 VkSamplerMipmapMode mipmap_mode)
    : core_(core) {
  VkSamplerCreateInfo sampler_create_info{};
  sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_create_info.magFilter = mag_filter;

  sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  sampler_create_info.magFilter = mag_filter;
  sampler_create_info.minFilter = min_filter;
  sampler_create_info.addressModeU = address_mode_u;
  sampler_create_info.addressModeV = address_mode_v;
  sampler_create_info.addressModeW = address_mode_w;
  sampler_create_info.anisotropyEnable = anisotropy_enable;
  sampler_create_info.maxAnisotropy = 1.0f;
  sampler_create_info.borderColor = border_color;
  sampler_create_info.unnormalizedCoordinates = VK_FALSE;
  sampler_create_info.compareEnable = VK_FALSE;
  sampler_create_info.compareOp = VK_COMPARE_OP_ALWAYS;
  sampler_create_info.mipmapMode = mipmap_mode;

  if (vkCreateSampler(core_->Device()->Handle(), &sampler_create_info, nullptr,
                      &sampler_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create texture sampler!");
  }
}

Sampler::~Sampler() {
  vkDestroySampler(core_->Device()->Handle(), sampler_, nullptr);
}

VkSampler Sampler::Handle() const {
  return sampler_;
}

struct Core *Sampler::Core() const {
  return core_;
}

}  // namespace grassland::vulkan
