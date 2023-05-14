#pragma once

#include "descriptor_set_layout.h"

namespace grassland::vulkan {
class PipelineLayout {
 public:
  GRASSLAND_CANNOT_COPY(PipelineLayout)
  PipelineLayout(const class Device &device,
                 const DescriptorSetLayout &descriptor_set_layout);
  PipelineLayout(
      const class Device &device,
      const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts);
  ~PipelineLayout();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkPipelineLayout, layout_)
};
}  // namespace grassland::vulkan
