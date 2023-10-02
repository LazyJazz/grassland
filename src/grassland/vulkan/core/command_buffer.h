#pragma once
#include "grassland/vulkan/core/command_pool.h"

namespace grassland::vulkan {
class CommandBuffer {
 public:
  explicit CommandBuffer(class CommandPool *command_pool);
  ~CommandBuffer();

  [[nodiscard]] VkCommandBuffer Handle() const;
  [[nodiscard]] class CommandPool *CommandPool() const;

 private:
  class CommandPool *command_pool_;
  VkCommandBuffer command_buffer_{};
};
}  // namespace grassland::vulkan
