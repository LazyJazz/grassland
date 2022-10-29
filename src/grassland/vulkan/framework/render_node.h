#pragma once
#include <grassland/vulkan/framework/core.h>
#include <grassland/vulkan/framework/render_node_settings.h>
#include <grassland/vulkan/framework/texture_image.h>
#include <grassland/vulkan/framework/uniform_bindings.h>

namespace grassland::vulkan::framework {
class RenderNode {
 public:
  void AddUniformBindingBuffer(Buffer *uniform_buffer,
                               VkPipelineStageFlags access_stage_flag);
  void AddUniformBindingTextureSampler(TextureImage *texture_image,
                                       Sampler *sampler,
                                       VkPipelineStageFlags access_stage_flag);
  void AddUniformBindingTextureSamplers(
      std::vector<std::pair<TextureImage *, Sampler *>> texture_sampler_pairs_,
      VkPipelineStageFlags access_stage_flag);
  void AddUniformBindingStorageTexture(TextureImage *texture_,
                                       VkPipelineStageFlags access_stage_flag);
  void AddUniformBindingStorageTextures(std::vector<TextureImage *> textures_,
                                        VkPipelineStageFlags access_stage_flag);
  int AddColorAttachment(VkFormat format,
                         VkPipelineColorBlendAttachmentState blend_state);
  void BuildRenderNode();
  void Draw(Buffer *vertex_buffer, Buffer *index_buffer);

 private:
  std::unique_ptr<RenderPass> render_pass_;
  std::unique_ptr<PipelineLayout> pipeline_layout_;
  std::unique_ptr<DescriptorPool> descriptor_pool_;
  std::vector<std::unique_ptr<DescriptorSet>> descriptor_sets_;
  RenderNodeSettings render_node_settings_;
  const std::string shader_code_uniform_list_;
};
}  // namespace grassland::vulkan::framework
