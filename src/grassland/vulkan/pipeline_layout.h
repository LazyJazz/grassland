#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class PipelineLayout {
 public:
  explicit PipelineLayout(Device *device);
  ~PipelineLayout();

 private:
  GRASSLAND_VULKAN_HANDLE(VkPipelineLayout)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
