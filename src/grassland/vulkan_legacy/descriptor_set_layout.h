#pragma once
#include <grassland/vulkan_legacy/device.h>
#include <grassland/vulkan_legacy/helper/descriptor_set_layout_bindings.h>

namespace grassland::vulkan_legacy {
class DescriptorSetLayout {
 public:
  DescriptorSetLayout(
      Device *device,
      const helper::DescriptorSetLayoutBindings &layout_bindings);
  DescriptorSetLayout(
      Device *device,
      const std::vector<VkDescriptorSetLayoutBinding> &layout_bindings);
  ~DescriptorSetLayout();

 private:
  GRASSLAND_VULKAN_HANDLE(VkDescriptorSetLayout)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
