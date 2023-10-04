#pragma once
#include <grassland/vulkan_legacy/buffer.h>
#include <grassland/vulkan_legacy/descriptor_set.h>
#include <grassland/vulkan_legacy/image.h>
#include <grassland/vulkan_legacy/image_view.h>
#include <grassland/vulkan_legacy/sampler.h>

namespace grassland::vulkan_legacy::helper {
void UpdateDescriptorWrite(VkDevice &device,
                           VkDescriptorSet &descriptor_set,
                           int binding,
                           Buffer *buffer);
void UpdateDescriptorWrite(VkDevice &device,
                           VkDescriptorSet &descriptor_set,
                           int binding,
                           ImageView *image_view,
                           Sampler *sampler);
}  // namespace grassland::vulkan_legacy::helper
