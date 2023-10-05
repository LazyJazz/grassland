#pragma once

#include "grassland/vulkan/core/core.h"
#include "grassland/vulkan/descriptors/descriptor_set_layout.h"

namespace grassland::vulkan {

struct DescriptorPoolSettings {
  // Give the maximum number of descriptor sets that can be allocated from the
  // pool.
  uint32_t max_sets{1024};
  uint32_t num_combined_image_samplers{1024};
  uint32_t num_uniform_buffers{1024};
  uint32_t num_storage_buffers{1024};
  uint32_t num_uniform_texel_buffers{1024};
  uint32_t num_storage_texel_buffers{1024};
  uint32_t num_sampled_images{1024};
  uint32_t num_storage_images{1024};
  uint32_t num_uniform_buffers_dynamic{1024};
  uint32_t num_storage_buffers_dynamic{1024};
  uint32_t num_input_attachments{1024};
  uint32_t num_samplers{1024};
  uint32_t num_acceleration_structures{0};
};

class DescriptorPool {
 public:
  DescriptorPool(class Core *core,
                 const std::vector<VkDescriptorPoolSize> &pool_sizes,
                 uint32_t max_sets);
  explicit DescriptorPool(
      class Core *core,
      const DescriptorPoolSettings &settings = DescriptorPoolSettings());
  DescriptorPool(class Core *core,
                 const std::vector<VkDescriptorSetLayoutBinding> &bindings,
                 uint32_t max_sets);
  DescriptorPool(class Core *core,
                 DescriptorSetLayout *layout,
                 uint32_t max_sets);
  ~DescriptorPool();

  [[nodiscard]] VkDescriptorPool Handle() const;
  [[nodiscard]] class Core *Core() const;

 private:
  class Core *core_{};
  VkDescriptorPool pool_{};
};

}  // namespace grassland::vulkan
