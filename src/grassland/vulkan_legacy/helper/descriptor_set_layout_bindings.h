#pragma once
#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy::helper {
class DescriptorSetLayoutBindings {
 public:
  void AddBinding(VkDescriptorSetLayoutBinding binding);
  void AddBinding(uint32_t binding,
                  VkDescriptorType descriptorType,
                  uint32_t descriptorCount,
                  VkShaderStageFlags stageFlags,
                  const VkSampler *pImmutableSamplers = nullptr);
  [[nodiscard]] const std::vector<VkDescriptorSetLayoutBinding>
      &GetDescriptorSetLayoutBinding() const;

 private:
  std::vector<VkDescriptorSetLayoutBinding> bindings_;
};
}  // namespace grassland::vulkan_legacy::helper
