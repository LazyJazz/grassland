#pragma once
#include <grassland/vulkan/framework/core.h>
#include <grassland/vulkan/framework/data_buffer.h>
#include <grassland/vulkan/framework/texture_image.h>
#include <grassland/vulkan/framework/uniform_bindings.h>

namespace grassland::vulkan::framework {
class RenderNode {
 public:
  explicit RenderNode(Core *core);
  void AddUniformBinding(UniformBinding *uniform_binding);
  int ColorOutput(VkFormat format,
                  VkPipelineColorBlendAttachmentState blend_state);
  void EnableDepthTest();
  void BuildRenderNode(uint32_t width, uint32_t height);
  void Draw(DataBuffer *vertex_buffer, DataBuffer *index_buffer);
  void AddShader(const char *shader_path, VkShaderStageFlags shader_stage);

  [[nodiscard]] TextureImage *GetColorImage(int color_image_index = 0) const;
  [[nodiscard]] TextureImage *GetDepthImage() const;

 private:
  Core *core_{nullptr};
  std::unique_ptr<PipelineLayout> pipeline_layout_;
  std::unique_ptr<DescriptorPool> descriptor_pool_;
  std::vector<std::unique_ptr<DescriptorSet>> descriptor_sets_;
  std::vector<std::unique_ptr<UniformBinding>> uniform_bindings_;

  std::unique_ptr<RenderPass> render_pass_;
  std::vector<std::unique_ptr<TextureImage>> color_attachment_textures_;
  std::unique_ptr<TextureImage> depth_buffer_texture_;
  helper::AttachmentParameters attachment_parameters_;

  std::vector<ShaderModule> shader_modules_;
  helper::ShaderStages shader_stages_;

  std::string shader_code_vertex_input_list_;
  std::string shader_code_uniform_list_;
  std::string shader_code_output_list_;
};
}  // namespace grassland::vulkan::framework
