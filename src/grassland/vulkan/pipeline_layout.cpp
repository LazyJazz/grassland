#include <grassland/logging/logging.h>
#include <grassland/vulkan/pipeline_layout.h>

namespace grassland::vulkan {
PipelineLayout::PipelineLayout(Device *device) : handle_{} {
  device_ = device;
  VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 0;
  pipelineLayoutInfo.pushConstantRangeCount = 0;

  if (vkCreatePipelineLayout(device->GetHandle(), &pipelineLayoutInfo, nullptr,
                             &handle_) != VK_SUCCESS) {
    LAND_ERROR("Vulkan failed to create pipeline layout!");
  }
}

PipelineLayout::~PipelineLayout() {
  vkDestroyPipelineLayout(device_->GetHandle(), handle_, nullptr);
}
}  // namespace grassland::vulkan
