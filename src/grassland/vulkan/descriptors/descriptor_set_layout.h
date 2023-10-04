#pragma once

#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {

struct DescriptorSetLayout {
 public:
  DescriptorSetLayout(
      class Core *core,
      const std::vector<VkDescriptorSetLayoutBinding> &bindings);
  ~DescriptorSetLayout();

  [[nodiscard]] VkDescriptorSetLayout Handle() const;
  [[nodiscard]] class Core *Core() const;

  [[nodiscard]] const std::vector<VkDescriptorSetLayoutBinding> &Bindings()
      const {
    return bindings_;
  }

 private:
  class Core *core_{};
  VkDescriptorSetLayout layout_{};

  std::vector<VkDescriptorSetLayoutBinding> bindings_{};
};

}  // namespace grassland::vulkan
