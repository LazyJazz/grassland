#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class PipelineLayout {
 public:
  explicit PipelineLayout(Device *device);
  ~PipelineLayout();

 private:
  VK_HANDLE(VkPipelineLayout)
  VK_DEVICE_PTR
};
}  // namespace grassland::vulkan
