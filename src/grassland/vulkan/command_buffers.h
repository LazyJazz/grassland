#pragma once
#include "command_pool.h"

namespace grassland::vulkan {
class CommandBuffers {
 public:
  GRASSLAND_CANNOT_COPY(CommandBuffers);
  CommandBuffers(const CommandPool &command_pool, uint32_t size);
  ~CommandBuffers();
  [[nodiscard]] uint32_t Size() const {
    return command_buffers_.size();
  }
  VkCommandBuffer &operator[](const size_t i) {
    return command_buffers_[i];
  }

  VkCommandBuffer Begin(size_t i = 0);
  void End(size_t i = 0);

 private:
  const CommandPool &command_pool_;
  std::vector<VkCommandBuffer> command_buffers_{};
};
}  // namespace grassland::vulkan
