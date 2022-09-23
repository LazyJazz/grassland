#pragma once
#include <grassland/vulkan/command_pool.h>
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/queue.h>

namespace grassland::vulkan {
class Buffer {
 public:
  Buffer(
      Device *device,
      VkDeviceSize size,
      VkBufferUsageFlags usage,
      VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  ~Buffer();

  void *Map(uint64_t size = 0xffffffffffffffffull, uint64_t offset = 0);
  void Unmap();
  [[nodiscard]] VkDeviceMemory GetDeviceMemory() const;
  [[nodiscard]] VkDeviceSize Size() const;
  void UploadData(Queue *graphics_queue,
                  CommandPool *command_pool,
                  const void *src_data,
                  VkDeviceSize size = 0xffffffffffffffffull,
                  VkDeviceSize offset = 0);
  void RetrieveData(Queue *graphics_queue,
                    CommandPool *command_pool,
                    void *dst_data,
                    VkDeviceSize size = 0xffffffffffffffffull,
                    VkDeviceSize offset = 0);

 private:
  GRASSLAND_VULKAN_HANDLE(VkBuffer)
  GRASSLAND_VULKAN_DEVICE_PTR
  VkDeviceMemory device_memory_{};
  VkDeviceSize size_{};
};

void CopyBuffer(Queue *graphics_queue,
                CommandPool *command_pool,
                VkBuffer src_buffer,
                VkBuffer dst_buffer,
                VkDeviceSize size,
                VkDeviceSize src_offset,
                VkDeviceSize dst_offset);

}  // namespace grassland::vulkan