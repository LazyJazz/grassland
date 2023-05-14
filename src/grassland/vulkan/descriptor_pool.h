#pragma once
#include "device.h"

namespace grassland::vulkan {
class DescriptorPool {
 public:
  GRASSLAND_CANNOT_COPY(DescriptorPool)
  DescriptorPool(const class Device &device,
                 const std::vector<VkDescriptorSetLayoutBinding> &bindings,
                 size_t max_sets);
  DescriptorPool(const class Device &device,
                 const std::vector<VkDescriptorPoolSize> &pool_sizes,
                 size_t max_sets);
  ~DescriptorPool();

 private:
  void CreatePool(const std::vector<VkDescriptorPoolSize> &pool_sizes,
                  size_t max_sets);

  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkDescriptorPool, descriptor_pool_)
};
}  // namespace grassland::vulkan
