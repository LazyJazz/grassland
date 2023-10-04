#include "grassland/vulkan/descriptors/descriptor_set.h"

namespace grassland::vulkan {

DescriptorSet::DescriptorSet(class Core *core,
                             DescriptorPool *descriptor_pool,
                             DescriptorSetLayout *descriptor_set_layout)
    : core_(core), descriptor_pool_(descriptor_pool) {
  auto layout = descriptor_set_layout->Handle();
  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool->Handle();
  alloc_info.descriptorSetCount = 1;
  alloc_info.pSetLayouts = &layout;

  if (vkAllocateDescriptorSets(core_->Device()->Handle(), &alloc_info, &set_) !=
      VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to allocate descriptor set!");
  }
}

DescriptorSet::~DescriptorSet() {
  vkFreeDescriptorSets(core_->Device()->Handle(), descriptor_pool_->Handle(), 1,
                       &set_);
}

VkDescriptorSet DescriptorSet::Handle() const {
  return set_;
}

struct Core *DescriptorSet::Core() const {
  return core_;
}

DescriptorPool *DescriptorSet::Pool() const {
  return descriptor_pool_;
}
}  // namespace grassland::vulkan
