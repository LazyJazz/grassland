#pragma once
#include "device_memory.h"

namespace grassland::vulkan {
class Buffer {
 public:
  GRASSLAND_CANNOT_COPY(Buffer)
  Buffer(const class Device &device,
         size_t size,
         VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                                    VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  ~Buffer();
  DeviceMemory AllocateMemory(VkMemoryPropertyFlags property_flags =
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  DeviceMemory AllocateMemory(VkMemoryAllocateFlags allocate_flags,
                              VkMemoryPropertyFlags property_flags =
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  [[nodiscard]] VkDeviceAddress GetDeviceAddress() const;

 private:
  [[nodiscard]] VkMemoryRequirements GetMemoryRequirements() const;

  GRASSLAND_VULKAN_HANDLE(VkBuffer, buffer_)
  GRASSLAND_VULKAN_DEVICE
};
}  // namespace grassland::vulkan
