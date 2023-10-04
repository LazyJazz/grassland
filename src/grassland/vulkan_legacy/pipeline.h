#pragma once
#include <grassland/vulkan_legacy/helper/shader_stages.h>
#include <grassland/vulkan_legacy/helper/vertex_input_descriptions.h>
#include <grassland/vulkan_legacy/pipeline_layout.h>
#include <grassland/vulkan_legacy/render_pass.h>
#include <grassland/vulkan_legacy/shader_module.h>

#include "grassland/vulkan/core/util.h"

namespace grassland::vulkan_legacy {
class Pipeline {
 public:
  explicit Pipeline(
      Device *device,
      RenderPass *render_pass,
      PipelineLayout *pipeline_layout,
      const helper::ShaderStages &shader_stages,
      const helper::VertexInputDescriptions &vertex_input_descriptions,
      bool depth_test = false,
      bool blend_enable = false);
  Pipeline(Device *device,
           RenderPass *render_pass,
           PipelineLayout *pipeline_layout,
           const helper::ShaderStages &shader_stages,
           const helper::VertexInputDescriptions &vertex_input_descriptions,
           const std::vector<VkPipelineColorBlendAttachmentState>
               &pipeline_color_blend_attachment_states,
           bool depth_test = false);
  ~Pipeline();

 private:
  GRASSLAND_VULKAN_HANDLE(VkPipeline)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan_legacy
