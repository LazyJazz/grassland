#include "descriptor_set_layout.h"

namespace grassland::vulkan {
DescriptorSetLayout::DescriptorSetLayout(
    const class Device &device,
    const std::vector<VkDescriptorSetLayoutBinding> &bindings)
    : device_(device) {
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = bindings.size();
  layout_info.pBindings = bindings.data();

  GRASSLAND_VULKAN_CHECK(vkCreateDescriptorSetLayout(
      device_.Handle(), &layout_info, nullptr, &layout_));
}

DescriptorSetLayout::~DescriptorSetLayout() {
  if (layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(device_.Handle(), layout_, nullptr);
  }
}

}  // namespace grassland::vulkan
