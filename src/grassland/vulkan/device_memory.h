#pragma once
#include "device.h"

namespace grassland::vulkan {
class DeviceMemory {
 public:
  DeviceMemory(const DeviceMemory &) = delete;
  DeviceMemory &operator=(const DeviceMemory &) = delete;
  DeviceMemory &operator=(DeviceMemory &&) = delete;

  DeviceMemory(const class Device &device,
               size_t size,
               uint32_t memory_type_bits,
               VkMemoryAllocateFlags allocate_flags =
                   VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
               VkMemoryPropertyFlags property_flags =
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  DeviceMemory(DeviceMemory &&device_memory) noexcept;
  ~DeviceMemory();

  void *Map(size_t offset, size_t size);
  void Unmap();

 private:
  [[nodiscard]] uint32_t FindMemoryType(
      uint32_t type_filter,
      VkMemoryPropertyFlags property_flags) const;

  GRASSLAND_VULKAN_HANDLE(VkDeviceMemory, memory_)
  GRASSLAND_VULKAN_DEVICE
};
}  // namespace grassland::vulkan
