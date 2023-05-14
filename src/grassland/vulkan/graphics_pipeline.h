#pragma once
#include "pipeline_layout.h"
#include "render_pass.h"

namespace grassland::vulkan {
class Pipeline {
 public:
  GRASSLAND_CANNOT_COPY(Pipeline);
  Pipeline(
      const RenderPass &render_pass,
      const Pipeline &pipeline,
      const std::vector<VkPipelineShaderStageCreateInfo> &shader_stage_infos,
      const std::vector<VkVertexInputBindingDescription> &binding_descriptions,
      const std::vector<VkVertexInputAttributeDescription>
          &attribute_descriptions, );
  ~Pipeline();

 private:
  GRASSLAND_VULKAN_DEVICE
  GRASSLAND_VULKAN_HANDLE(VkPipeline, pipeline_)
};
}  // namespace grassland::vulkan
