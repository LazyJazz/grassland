#pragma once
#include "device.h"

namespace grassland::vulkan {
class DescriptorSetLayout {
 public:
  GRASSLAND_CANNOT_COPY(DescriptorSetLayout)

  DescriptorSetLayout(
      const class Device &device,
      const std::vector<VkDescriptorSetLayoutBinding> &bindings);
  ~DescriptorSetLayout();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkDescriptorSetLayout, layout_)
};
}  // namespace grassland::vulkan
