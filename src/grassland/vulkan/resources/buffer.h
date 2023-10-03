#pragma once
#include "grassland/vulkan/core/core.h"

namespace grassland::vulkan {
class Buffer {
 public:
  Buffer(class Core *core,
         VkDeviceSize size,
         VkBufferUsageFlags usage,
         VmaMemoryUsage memory_usage);
  ~Buffer();

  [[nodiscard]] class Core *Core() const;

  [[nodiscard]] VkBuffer Handle() const;

  [[nodiscard]] VmaAllocation Allocation() const;

  [[nodiscard]] VkDeviceSize Size() const;

  [[nodiscard]] void *Map() const;
  void Unmap() const;

 private:
  class Core *core_{};
  VkBuffer buffer_{};
  VkDeviceSize size_{};
  VmaAllocation allocation_{};
};
}  // namespace grassland::vulkan
