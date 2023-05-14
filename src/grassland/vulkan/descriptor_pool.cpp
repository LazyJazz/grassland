#include "descriptor_pool.h"

namespace grassland::vulkan {

DescriptorPool::DescriptorPool(
    const class Device &device,
    const std::vector<VkDescriptorPoolSize> &pool_sizes,
    size_t max_sets)
    : device_(device) {
  CreatePool(pool_sizes, max_sets);
}

DescriptorPool::DescriptorPool(
    const class Device &device,
    const std::vector<VkDescriptorSetLayoutBinding> &bindings,
    size_t max_sets)
    : device_(device) {
  std::vector<VkDescriptorPoolSize> pool_sizes;
  for (auto &binding : bindings) {
    VkDescriptorPoolSize pool_size;
    pool_size.descriptorCount = binding.descriptorCount * max_sets;
    pool_size.type = binding.descriptorType;
    pool_sizes.push_back(pool_size);
  }
  CreatePool(pool_sizes, max_sets);
}

DescriptorPool::~DescriptorPool() {
  if (descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device_.Handle(), descriptor_pool_, nullptr);
  }
}

void DescriptorPool::CreatePool(
    const std::vector<VkDescriptorPoolSize> &pool_sizes,
    size_t max_sets) {
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = max_sets;
  GRASSLAND_VULKAN_CHECK(vkCreateDescriptorPool(device_.Handle(), &pool_info,
                                                nullptr, &descriptor_pool_));
}

}  // namespace grassland::vulkan
