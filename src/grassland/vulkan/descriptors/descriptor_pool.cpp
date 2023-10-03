#include "grassland/vulkan/descriptors/descriptor_pool.h"

namespace grassland::vulkan {

namespace {
std::vector<VkDescriptorPoolSize> VectorPoolSizesFromBindings(
    const std::vector<VkDescriptorSetLayoutBinding> &bindings,
    uint32_t max_sets) {
  std::vector<VkDescriptorPoolSize> pool_sizes{};
  for (auto binding : bindings) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = binding.descriptorType;
    pool_size.descriptorCount =
        static_cast<uint32_t>(binding.descriptorCount * max_sets);
    pool_sizes.push_back(pool_size);
  }
  return pool_sizes;
}

std::vector<VkDescriptorPoolSize> VectorPoolSizesFromSettings(
    const DescriptorPoolSettings &settings) {
  std::vector<VkDescriptorPoolSize> pool_sizes{};
  if (settings.num_combined_image_samplers > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = settings.num_combined_image_samplers;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_uniform_buffers > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = settings.num_uniform_buffers;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_storage_buffers > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = settings.num_storage_buffers;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_uniform_texel_buffers > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    pool_size.descriptorCount = settings.num_uniform_texel_buffers;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_storage_texel_buffers > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER;
    pool_size.descriptorCount = settings.num_storage_texel_buffers;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_sampled_images > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_size.descriptorCount = settings.num_sampled_images;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_storage_images > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_size.descriptorCount = settings.num_storage_images;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_uniform_buffers_dynamic > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    pool_size.descriptorCount = settings.num_uniform_buffers_dynamic;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_storage_buffers_dynamic > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    pool_size.descriptorCount = settings.num_storage_buffers_dynamic;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_input_attachments > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
    pool_size.descriptorCount = settings.num_input_attachments;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_samplers > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLER;
    pool_size.descriptorCount = settings.num_samplers;
    pool_sizes.push_back(pool_size);
  }
  if (settings.num_acceleration_structures > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV;
    pool_size.descriptorCount = settings.num_acceleration_structures;
    pool_sizes.push_back(pool_size);
  }
  return pool_sizes;
}
}  // namespace

DescriptorPool::DescriptorPool(
    struct Core *core,
    const std::vector<VkDescriptorPoolSize> &pool_sizes,
    uint32_t max_sets)
    : core_(core) {
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.poolSizeCount = pool_sizes.size();
  pool_info.pPoolSizes = pool_sizes.data();
  pool_info.maxSets = max_sets;
  pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

  if (vkCreateDescriptorPool(core_->Device()->Handle(), &pool_info, nullptr,
                             &pool_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create descriptor pool!");
  }
}

DescriptorPool::DescriptorPool(struct Core *core,
                               const DescriptorPoolSettings &settings)
    : DescriptorPool(core,
                     VectorPoolSizesFromSettings(settings),
                     settings.max_sets) {
}

DescriptorPool::DescriptorPool(
    struct Core *core,
    const std::vector<VkDescriptorSetLayoutBinding> &bindings,
    uint32_t max_sets)
    : DescriptorPool(core,
                     VectorPoolSizesFromBindings(bindings, max_sets),
                     max_sets) {
}

DescriptorPool::DescriptorPool(struct Core *core,
                               DescriptorSetLayout *layout,
                               uint32_t max_sets)
    : DescriptorPool(core, layout->Bindings(), max_sets) {
}

DescriptorPool::~DescriptorPool() {
  vkDestroyDescriptorPool(core_->Device()->Handle(), pool_, nullptr);
}

VkDescriptorPool DescriptorPool::Handle() const {
  return pool_;
}

class Core *DescriptorPool::Core() const {
  return core_;
}

}  // namespace grassland::vulkan
