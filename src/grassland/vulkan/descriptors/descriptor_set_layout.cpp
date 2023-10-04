#include "grassland/vulkan/descriptors/descriptor_set_layout.h"

namespace grassland::vulkan {

DescriptorSetLayout::DescriptorSetLayout(
    class Core *core,
    const std::vector<VkDescriptorSetLayoutBinding> &bindings)
    : core_(core), bindings_(bindings) {
  // Create the descriptor set layout
  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.data();

  if (vkCreateDescriptorSetLayout(core_->Device()->Handle(), &layout_info,
                                  nullptr, &layout_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create descriptor set layout!");
  }
}

DescriptorSetLayout::~DescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(core_->Device()->Handle(), layout_, nullptr);
}

VkDescriptorSetLayout DescriptorSetLayout::Handle() const {
  return layout_;
}

class Core *DescriptorSetLayout::Core() const {
  return core_;
}

}  // namespace grassland::vulkan
