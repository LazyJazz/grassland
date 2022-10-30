#pragma once
#include <grassland/vulkan/framework/core.h>
#include <grassland/vulkan/framework/data_buffer.h>
#include <grassland/vulkan/framework/texture_image.h>
#include <grassland/vulkan/framework/uniform_bindings.h>

namespace grassland::vulkan::framework {
class RenderNode {
 public:
  explicit RenderNode(Core *core);
  void AddUniformBinding(DataBuffer *uniform_buffer,
                         VkShaderStageFlags access_stage_flags);
  void AddUniformBinding(TextureImage *texture_image,
                         Sampler *sampler,
                         VkShaderStageFlags access_stage_flags);
  void AddUniformBinding(const std::vector<std::pair<TextureImage *, Sampler *>>
                             &texture_sampler_pairs,
                         VkShaderStageFlags access_stage_flags);
  void AddUniformBinding(TextureImage *texture_image,
                         VkShaderStageFlags access_stage_flags);
  void AddUniformBinding(const std::vector<TextureImage *> &texture_images,
                         VkShaderStageFlags access_stage_flags);
  int AddColorOutput(VkFormat format,
                     VkPipelineColorBlendAttachmentState blend_state);
  int AddColorOutput(VkFormat format, bool blend_enable = false);
  void EnableDepthTest();

  void AddShader(const char *shader_path, VkShaderStageFlagBits shader_stage);

  void VertexInput(const std::vector<VkFormat> &vertex_inputs);

  void BuildRenderNode(uint32_t width, uint32_t height);

  void Draw(DataBuffer *vertex_buffer,
            DataBuffer *index_buffer,
            int index_count,
            int instance_index);
  void Draw(CommandBuffer *command_buffer,
            DataBuffer *vertex_buffer,
            DataBuffer *index_buffer,
            int index_count,
            int instance_index);
  void Draw(VkCommandBuffer command_buffer,
            DataBuffer *vertex_buffer,
            DataBuffer *index_buffer,
            int index_count,
            int instance_index);

  [[nodiscard]] TextureImage *GetColorImage(int color_image_index = 0) const;
  [[nodiscard]] TextureImage *GetDepthImage() const;

 private:
  Core *core_{nullptr};

  std::unique_ptr<DescriptorSetLayout> descriptor_set_layout_;
  std::unique_ptr<PipelineLayout> pipeline_layout_;
  std::vector<std::unique_ptr<UniformBinding>> uniform_bindings_;
  std::unique_ptr<DescriptorPool> descriptor_pool_;
  std::vector<std::unique_ptr<DescriptorSet>> descriptor_sets_;

  std::unique_ptr<RenderPass> render_pass_;
  std::unique_ptr<Framebuffer> framebuffer_;
  std::vector<std::unique_ptr<TextureImage>> color_attachment_textures_;
  std::unique_ptr<TextureImage> depth_buffer_texture_;
  std::vector<std::pair<VkFormat, VkPipelineColorBlendAttachmentState>>
      color_attachments_;
  bool depth_enable_{false};

  std::vector<std::unique_ptr<ShaderModule>> shader_modules_;
  helper::ShaderStages shader_stages_;

  helper::VertexInputDescriptions vertex_input_descriptions_;

  std::unique_ptr<Pipeline> graphics_pipeline_;
};
}  // namespace grassland::vulkan::framework
