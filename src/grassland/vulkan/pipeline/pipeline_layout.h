#pragma once
#include "grassland/vulkan/core/core.h"
#include "grassland/vulkan/descriptors/descriptor_set_layout.h"

namespace grassland::vulkan {
class PipelineLayout {
 public:
  PipelineLayout(
      class Core *core,
      const std::vector<DescriptorSetLayout *> &descriptor_set_layouts);
  PipelineLayout(
      class Core *core,
      const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts);
  ~PipelineLayout();

  [[nodiscard]] VkPipelineLayout Handle() const;
  [[nodiscard]] class Core *Core() const;

 private:
  class Core *core_{};

  VkPipelineLayout pipeline_layout_{};
};
}  // namespace grassland::vulkan
