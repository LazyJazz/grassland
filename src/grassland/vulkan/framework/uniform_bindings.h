#pragma once
#include <grassland/vulkan/framework/core.h>
#include <grassland/vulkan/framework/data_buffer.h>
#include <grassland/vulkan/framework/texture_image.h>

namespace grassland::vulkan::framework {
class UniformBinding {
 public:
  explicit UniformBinding(VkShaderStageFlags access_stage_flags);
  virtual ~UniformBinding() = default;
  [[nodiscard]] virtual VkDescriptorSetLayoutBinding GetBinding() const = 0;
  [[nodiscard]] virtual VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const = 0;
  virtual void PrepareState(CommandBuffer *command_buffer,
                            int frame_index) const = 0;

 protected:
  VkShaderStageFlags access_stage_flags_;
};

class UniformBindingUniform : public UniformBinding {
 public:
  explicit UniformBindingUniform(DataBuffer *uniform_buffer,
                                 VkShaderStageFlags access_stage_flags);
  ~UniformBindingUniform() override = default;

 private:
  [[nodiscard]] VkDescriptorSetLayoutBinding GetBinding() const override;
  [[nodiscard]] VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const override;
  void PrepareState(CommandBuffer *command_buffer,
                    int frame_index) const override;

  DataBuffer *uniform_buffer_{nullptr};
  std::vector<VkDescriptorBufferInfo> buffer_infos_;
};

class UniformBindingBuffer : public UniformBinding {
 public:
  explicit UniformBindingBuffer(DataBuffer *uniform_buffer,
                                VkShaderStageFlags access_stage_flags);
  ~UniformBindingBuffer() override = default;

 private:
  [[nodiscard]] VkDescriptorSetLayoutBinding GetBinding() const override;
  [[nodiscard]] VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const override;
  void PrepareState(CommandBuffer *command_buffer,
                    int frame_index) const override;

  DataBuffer *uniform_buffer_{nullptr};
  std::vector<VkDescriptorBufferInfo> buffer_infos_;
};

class UniformBindingTextureSampler : public UniformBinding {
 public:
  UniformBindingTextureSampler(TextureImage *texture_image,
                               Sampler *sampler,
                               VkShaderStageFlags access_stage_flags);
  ~UniformBindingTextureSampler() override = default;

 private:
  [[nodiscard]] VkDescriptorSetLayoutBinding GetBinding() const override;
  [[nodiscard]] VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const override;
  void PrepareState(CommandBuffer *command_buffer,
                    int frame_index) const override;

  TextureImage *texture_image_{nullptr};
  Sampler *sampler_{nullptr};
  VkDescriptorImageInfo image_info_{};
};

class UniformBindingTextureSamplers : public UniformBinding {
 public:
  UniformBindingTextureSamplers(
      const std::vector<std::pair<TextureImage *, Sampler *>>
          &texture_sampler_pairs,
      VkShaderStageFlags access_stage_flags);
  ~UniformBindingTextureSamplers() override = default;

 private:
  [[nodiscard]] VkDescriptorSetLayoutBinding GetBinding() const override;
  [[nodiscard]] VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const override;
  void PrepareState(CommandBuffer *command_buffer,
                    int frame_index) const override;

  std::vector<std::pair<TextureImage *, Sampler *>> texture_sampler_pairs_;
  std::vector<VkDescriptorImageInfo> image_infos_;
};

class UniformBindingStorageTexture : public UniformBinding {
 public:
  UniformBindingStorageTexture(TextureImage *texture_image,
                               VkShaderStageFlags access_stage_flags);
  ~UniformBindingStorageTexture() override = default;

 private:
  [[nodiscard]] VkDescriptorSetLayoutBinding GetBinding() const override;
  [[nodiscard]] VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const override;
  void PrepareState(CommandBuffer *command_buffer,
                    int frame_index) const override;

  TextureImage *texture_image_{nullptr};
  VkDescriptorImageInfo image_info_{};
};

class UniformBindingStorageTextures : public UniformBinding {
 public:
  UniformBindingStorageTextures(
      const std::vector<TextureImage *> &texture_images,
      VkShaderStageFlags access_stage_flags);
  ~UniformBindingStorageTextures() override = default;

 private:
  [[nodiscard]] VkDescriptorSetLayoutBinding GetBinding() const override;
  [[nodiscard]] VkWriteDescriptorSet GetWriteDescriptorSet(
      int frame_index) const override;
  void PrepareState(CommandBuffer *command_buffer,
                    int frame_index) const override;

  std::vector<TextureImage *> texture_images_;
  std::vector<VkDescriptorImageInfo> image_infos_;
};
}  // namespace grassland::vulkan::framework
