#include "grassland/vulkan/pipeline/pipeline_layout.h"

namespace grassland::vulkan {

namespace {
std::vector<VkDescriptorSetLayout> ConvertDescriptorSetLayouts(
    const std::vector<DescriptorSetLayout *> &descriptor_set_layouts) {
  std::vector<VkDescriptorSetLayout> result;
  result.reserve(descriptor_set_layouts.size());
  for (const auto &descriptor_set_layout : descriptor_set_layouts) {
    result.push_back(descriptor_set_layout->Handle());
  }
  return result;
}
}  // namespace

PipelineLayout::PipelineLayout(
    struct Core *core,
    const std::vector<DescriptorSetLayout *> &descriptor_set_layouts)
    : PipelineLayout(core,
                     ConvertDescriptorSetLayouts(descriptor_set_layouts)) {
}

PipelineLayout::PipelineLayout(
    struct Core *core,
    const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts)
    : core_(core) {
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = descriptor_set_layouts.size();
  if (pipeline_layout_info.setLayoutCount) {
    pipeline_layout_info.pSetLayouts = descriptor_set_layouts.data();
  }
  pipeline_layout_info.pushConstantRangeCount = 0;
  if (vkCreatePipelineLayout(core_->Device()->Handle(), &pipeline_layout_info,
                             nullptr, &pipeline_layout_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create pipeline layout!");
  }
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(core_->Device()->Handle(), pipeline_layout_, nullptr);
}

VkPipelineLayout PipelineLayout::Handle() const {
  return pipeline_layout_;
}

class Core *PipelineLayout::Core() const {
  return core_;
}

}  // namespace grassland::vulkan
