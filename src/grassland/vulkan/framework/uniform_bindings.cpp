#include <grassland/util/util.h>
#include <grassland/vulkan/framework/uniform_bindings.h>

namespace grassland::vulkan::framework {
UniformBinding::UniformBinding(VkPipelineStageFlags access_stage_flags) {
  access_stage_flags_ = access_stage_flags;
}

UniformBindingUniform::UniformBindingUniform(
    DataBuffer *uniform_buffer,
    VkPipelineStageFlags access_stage_flags)
    : UniformBinding(access_stage_flags) {
  uniform_buffer_ = uniform_buffer;
  buffer_infos_.resize(
      uniform_buffer_->GetCore()->GetCoreSettings().frames_in_flight);
  for (int frame_index = 0; frame_index < buffer_infos_.size(); frame_index++) {
    auto &buffer_info = buffer_infos_[frame_index];
    buffer_info.buffer = uniform_buffer_->GetBuffer(frame_index)->GetHandle();
    buffer_info.range = uniform_buffer_->BufferSize();
    buffer_info.offset = 0;
  }
}

VkDescriptorSetLayoutBinding UniformBindingUniform::GetBinding() const {
  VkDescriptorSetLayoutBinding result{};
  result.descriptorCount = 1;
  result.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  result.stageFlags = access_stage_flags_;
  result.pImmutableSamplers = nullptr;
  return result;
}

VkWriteDescriptorSet UniformBindingUniform::GetWriteDescriptorSet(
    int frame_index) const {
  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = nullptr;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &buffer_infos_[frame_index];
  return descriptorWrite;
}

void UniformBindingUniform::PrepareState(CommandBuffer *command_buffer,
                                         int frame_index) const {
  uniform_buffer_->Sync(frame_index);
}

UniformBindingBuffer::UniformBindingBuffer(
    DataBuffer *uniform_buffer,
    VkPipelineStageFlags access_stage_flags)
    : UniformBinding(access_stage_flags) {
  uniform_buffer_ = uniform_buffer;
  buffer_infos_.resize(
      uniform_buffer_->GetCore()->GetCoreSettings().frames_in_flight);
  for (int frame_index = 0; frame_index < buffer_infos_.size(); frame_index++) {
    auto &buffer_info = buffer_infos_[frame_index];
    buffer_info.buffer = uniform_buffer_->GetBuffer(frame_index)->GetHandle();
    buffer_info.range = uniform_buffer_->BufferSize();
    buffer_info.offset = 0;
  }
}

VkDescriptorSetLayoutBinding UniformBindingBuffer::GetBinding() const {
  VkDescriptorSetLayoutBinding result{};
  result.descriptorCount = 1;
  result.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  result.stageFlags = access_stage_flags_;
  result.pImmutableSamplers = nullptr;
  return result;
}

VkWriteDescriptorSet UniformBindingBuffer::GetWriteDescriptorSet(
    int frame_index) const {
  VkWriteDescriptorSet descriptorWrite{};
  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = nullptr;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pBufferInfo = &buffer_infos_[frame_index];
  return descriptorWrite;
}

void UniformBindingBuffer::PrepareState(CommandBuffer *command_buffer,
                                        int frame_index) const {
  uniform_buffer_->Sync(frame_index);
}

UniformBindingTextureSampler::UniformBindingTextureSampler(
    TextureImage *texture_image,
    Sampler *sampler,
    VkShaderStageFlags access_stage_flags)
    : UniformBinding(access_stage_flags) {
  texture_image_ = texture_image;
  sampler_ = sampler;
  image_info_.imageView = texture_image_->GetImageView()->GetHandle();
  image_info_.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  image_info_.sampler = sampler_->GetHandle();
}

VkDescriptorSetLayoutBinding UniformBindingTextureSampler::GetBinding() const {
  VkDescriptorSetLayoutBinding result{};
  result.descriptorCount = 1;
  result.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  result.stageFlags = access_stage_flags_;
  result.pImmutableSamplers = nullptr;
  return result;
}

VkWriteDescriptorSet UniformBindingTextureSampler::GetWriteDescriptorSet(
    int frame_index) const {
  VkWriteDescriptorSet descriptorWrite{};

  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = nullptr;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pImageInfo = &image_info_;

  return descriptorWrite;
}

void UniformBindingTextureSampler::PrepareState(CommandBuffer *command_buffer,
                                                int frame_index) const {
  TransitImageLayout(command_buffer->GetHandle(),
                     texture_image_->GetImage()->GetHandle(),
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                     VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
}

UniformBindingTextureSamplers::UniformBindingTextureSamplers(
    const std::vector<std::pair<TextureImage *, Sampler *>>
        &texture_sampler_pairs,
    VkShaderStageFlags access_stage_flags)
    : UniformBinding(access_stage_flags) {
  texture_sampler_pairs_ = texture_sampler_pairs;
  image_infos_.resize(texture_sampler_pairs_.size());
  for (int i = 0; i < texture_sampler_pairs_.size(); i++) {
    auto &image_info = image_infos_[i];
    auto &texture_sampler_pair = texture_sampler_pairs_[i];
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_info.imageView =
        texture_sampler_pair.first->GetImageView()->GetHandle();
    image_info.sampler = texture_sampler_pair.second->GetHandle();
  }
}

VkDescriptorSetLayoutBinding UniformBindingTextureSamplers::GetBinding() const {
  VkDescriptorSetLayoutBinding result{};
  result.descriptorCount = texture_sampler_pairs_.size();
  result.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  result.stageFlags = access_stage_flags_;
  result.pImmutableSamplers = nullptr;
  return result;
}

VkWriteDescriptorSet UniformBindingTextureSamplers::GetWriteDescriptorSet(
    int frame_index) const {
  VkWriteDescriptorSet descriptorWrite{};

  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = nullptr;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  descriptorWrite.descriptorCount = image_infos_.size();
  descriptorWrite.pImageInfo = image_infos_.data();

  return descriptorWrite;
}

void UniformBindingTextureSamplers::PrepareState(CommandBuffer *command_buffer,
                                                 int frame_index) const {
  for (auto &texture_image_pair : texture_sampler_pairs_) {
    TransitImageLayout(command_buffer->GetHandle(),
                       texture_image_pair.first->GetImage()->GetHandle(),
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                       VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
                       VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
  }
}

UniformBindingStorageTexture::UniformBindingStorageTexture(
    TextureImage *texture_image,
    VkShaderStageFlags access_stage_flags)
    : UniformBinding(access_stage_flags) {
  texture_image_ = texture_image;
  image_info_.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  image_info_.imageView = texture_image_->GetImageView()->GetHandle();
  image_info_.sampler = nullptr;
}

VkDescriptorSetLayoutBinding UniformBindingStorageTexture::GetBinding() const {
  VkDescriptorSetLayoutBinding result{};
  result.descriptorCount = 1;
  result.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  result.stageFlags = access_stage_flags_;
  result.pImmutableSamplers = nullptr;
  return result;
}

VkWriteDescriptorSet UniformBindingStorageTexture::GetWriteDescriptorSet(
    int frame_index) const {
  VkWriteDescriptorSet descriptorWrite{};

  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = nullptr;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  descriptorWrite.descriptorCount = 1;
  descriptorWrite.pImageInfo = &image_info_;

  return descriptorWrite;
}

void UniformBindingStorageTexture::PrepareState(CommandBuffer *command_buffer,
                                                int frame_index) const {
  TransitImageLayout(
      command_buffer->GetHandle(), texture_image_->GetImage()->GetHandle(),
      VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
      VK_IMAGE_ASPECT_COLOR_BIT);
}

UniformBindingStorageTextures::UniformBindingStorageTextures(
    const std::vector<TextureImage *> &texture_images,
    VkShaderStageFlags access_stage_flags)
    : UniformBinding(access_stage_flags) {
  texture_images_ = texture_images;
  image_infos_.resize(texture_images_.size());
  for (int i = 0; i < image_infos_.size(); i++) {
    auto &image_info = image_infos_[i];
    auto &texture_image = texture_images_[i];
    image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_info.imageView = texture_image->GetImageView()->GetHandle();
    image_info.sampler = nullptr;
  }
}

VkDescriptorSetLayoutBinding UniformBindingStorageTextures::GetBinding() const {
  VkDescriptorSetLayoutBinding result{};
  result.descriptorCount = texture_images_.size();
  result.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  result.stageFlags = access_stage_flags_;
  result.pImmutableSamplers = nullptr;
  return result;
}

VkWriteDescriptorSet UniformBindingStorageTextures::GetWriteDescriptorSet(
    int frame_index) const {
  VkWriteDescriptorSet descriptorWrite{};

  descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  descriptorWrite.dstSet = nullptr;
  descriptorWrite.dstBinding = 0;
  descriptorWrite.dstArrayElement = 0;
  descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  descriptorWrite.descriptorCount = image_infos_.size();
  descriptorWrite.pImageInfo = image_infos_.data();

  return descriptorWrite;
}
void UniformBindingStorageTextures::PrepareState(CommandBuffer *command_buffer,
                                                 int frame_index) const {
  for (auto &texture_image : texture_images_) {
    TransitImageLayout(
        command_buffer->GetHandle(), texture_image->GetImage()->GetHandle(),
        VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
  }
}
}  // namespace grassland::vulkan::framework
