#pragma once
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
                    const char *vertex_shader_file_path,
                    const char *fragment_shader_file_path,
                    const char *geometry_shader_file_path = nullptr);
  explicit Pipeline(Device *device,
                    RenderPass *render_pass,
                    PipelineLayout *pipeline_layout,
                    const std::vector<uint8_t> &vertex_shader_spv_data,
                    const std::vector<uint8_t> &fragment_shader_spv_data,
                    const std::vector<uint8_t> &geometry_shader_spv_data = {});
  explicit Pipeline(Device *device,
                    RenderPass *render_pass,
                    PipelineLayout *pipeline_layout,
                    ShaderModule *vertex_shader,
                    ShaderModule *fragment_shader,
                    ShaderModule *geometry_shader = nullptr);
  ~Pipeline();

 private:
  void ConstructorCommon(Device *device,
                         RenderPass *render_pass,
                         PipelineLayout *pipeline_layout,
                         ShaderModule *vertex_shader,
                         ShaderModule *fragment_shader,
                         ShaderModule *geometry_shader);
  GRASSLAND_VULKAN_HANDLE(VkPipeline)
  GRASSLAND_VULKAN_DEVICE_PTR
};
}  // namespace grassland::vulkan
