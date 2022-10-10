﻿#include <grassland/util/logging.h>
#include <grassland/vulkan/descriptor_pool.h>

namespace grassland::vulkan {

DescriptorPool::DescriptorPool(
    Device *device,
    const helper::DescriptorSetLayoutBindings &bindings,
    int max_sets)
    : handle_{} {
  device_ = device;
  std::vector<VkDescriptorPoolSize> pool_sizes{};
  for (auto binding : bindings.GetDescriptorSetLayoutBinding()) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = binding.descriptorType;
    pool_size.descriptorCount =
        static_cast<uint32_t>(max_sets * binding.descriptorCount);
    pool_sizes.push_back(pool_size);
  }

  VkDescriptorPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  poolInfo.poolSizeCount = pool_sizes.size();
  poolInfo.pPoolSizes = pool_sizes.data();
  poolInfo.maxSets = static_cast<uint32_t>(max_sets);
  poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (vkCreateDescriptorPool(device_->GetHandle(), &poolInfo, nullptr,
                             &handle_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create descriptor pool!");
  }
}

DescriptorPool::~DescriptorPool() {
  vkDestroyDescriptorPool(device_->GetHandle(), GetHandle(), nullptr);
}

}  // namespace grassland::vulkan
