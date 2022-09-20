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
  Device *device_{nullptr};
};
}  // namespace grassland::vulkan
