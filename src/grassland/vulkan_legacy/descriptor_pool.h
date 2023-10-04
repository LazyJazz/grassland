#pragma once
#include <grassland/vulkan_legacy/device.h>
#include <grassland/vulkan_legacy/helper/descriptor_set_layout_bindings.h>

namespace grassland::vulkan_legacy {
class DescriptorPool {
 public:
  DescriptorPool(Device *device,
                 const helper::DescriptorSetLayoutBindings &bindings,
                 int max_sets);
  DescriptorPool(Device *device,
                 const std::vector<VkDescriptorSetLayoutBinding> &bindings,
                 int max_sets);
  ~DescriptorPool();

 private:
  GRASSLAND_VULKAN_HANDLE(VkDescriptorPool)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
