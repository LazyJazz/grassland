#pragma once
#include <grassland/vulkan_legacy/descriptor_set_layout.h>
#include <grassland/vulkan_legacy/device.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class PipelineLayout {
 public:
  explicit PipelineLayout(Device *device);
  explicit PipelineLayout(Device *device,
                          DescriptorSetLayout *descriptor_set_layout);
  explicit PipelineLayout(
      Device *device,
      const std::vector<VkDescriptorSetLayout> &descriptor_set_layouts);
  ~PipelineLayout();

 private:
  GRASSLAND_VULKAN_HANDLE(VkPipelineLayout)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
