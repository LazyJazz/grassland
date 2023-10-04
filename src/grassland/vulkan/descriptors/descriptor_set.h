#pragma once

#include "grassland/vulkan/core/core.h"
#include "grassland/vulkan/descriptors/descriptor_pool.h"
#include "grassland/vulkan/descriptors/descriptor_set_layout.h"

namespace grassland::vulkan {
class DescriptorSet {
 public:
  DescriptorSet(class Core *core,
                DescriptorPool *descriptor_pool,
                DescriptorSetLayout *descriptor_set_layout);
  ~DescriptorSet();

  [[nodiscard]] VkDescriptorSet Handle() const;
  [[nodiscard]] class Core *Core() const;
  [[nodiscard]] DescriptorPool *Pool() const;

 private:
  class Core *core_{};
  DescriptorPool *descriptor_pool_{};
  VkDescriptorSet set_{};
};
}  // namespace grassland::vulkan
