#pragma once
#include <grassland/vulkan/helper/shader_stages.h>
#include <grassland/vulkan/pipeline_layout.h>
#include <grassland/vulkan/render_pass.h>
#include <grassland/vulkan/shader_module.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Pipeline {
 public:
  explicit Pipeline(Device *device,
                    RenderPass *render_pass,
                    PipelineLayout *pipeline_layout,
                    const helper::ShaderStages &shader_stages);
  ~Pipeline();

 private:
  GRASSLAND_VULKAN_HANDLE(VkPipeline)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
