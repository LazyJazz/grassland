#include "pipeline_layout.h"

namespace grassland::vulkan {
PipelineLayout::PipelineLayout(
    const class Device &device,
    const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts)
    : device_(device) {
  VkPipelineLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout_info.setLayoutCount = descriptor_set_layouts.size();
  layout_info.pSetLayouts = descriptor_set_layouts.data();
  GRASSLAND_VULKAN_CHECK(vkCreatePipelineLayout(device_.Handle(), &layout_info,
                                                nullptr, &layout_));
}

PipelineLayout::PipelineLayout(const class Device &device,
                               const DescriptorSetLayout &descriptor_set_layout)
    : PipelineLayout(device, {descriptor_set_layout.Handle()}) {
}

PipelineLayout::~PipelineLayout() {
  if (layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(device_.Handle(), layout_, nullptr);
  }
}

}  // namespace grassland::vulkan
