#include "device_memory.h"

namespace grassland::vulkan {

DeviceMemory::DeviceMemory(const class Device &device,
                           size_t size,
                           uint32_t memory_type_bits,
                           VkMemoryAllocateFlags allocate_flags,
                           VkMemoryPropertyFlags property_flags)
    : device_(device) {
  VkMemoryAllocateFlagsInfo flags_info{};
  flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags_info.flags = allocate_flags;

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.pNext = &flags_info;
  alloc_info.allocationSize = size;
  alloc_info.memoryTypeIndex = FindMemoryType(memory_type_bits, property_flags);

  GRASSLAND_VULKAN_CHECK(
      vkAllocateMemory(device_.Handle(), &alloc_info, nullptr, &memory_));
}

DeviceMemory::DeviceMemory(DeviceMemory &&device_memory) noexcept
    : device_(device_memory.device_), memory_(device_memory.memory_) {
  device_memory.memory_ = VK_NULL_HANDLE;
}

uint32_t DeviceMemory::FindMemoryType(
    uint32_t type_filter,
    VkMemoryPropertyFlags property_flags) const {
  auto &mem_prop = device_.PhysicalDeviceMemoryProperties();

  for (uint32_t i = 0; i < mem_prop.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) && (mem_prop.memoryTypes[i].propertyFlags &
                                     property_flags) == property_flags) {
      return i;
    }
  }

  LAND_ERROR("Failed to find suitable memory type.");
}

DeviceMemory::~DeviceMemory() {
  if (memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_.Handle(), memory_, nullptr);
    memory_ = nullptr;
  }
}

void DeviceMemory::Unmap() {
  vkUnmapMemory(device_.Handle(), memory_);
}

void *DeviceMemory::Map(size_t offset, size_t size) {
  void *data = nullptr;
  GRASSLAND_VULKAN_CHECK(
      vkMapMemory(device_.Handle(), memory_, offset, size, 0, &data));
  return data;
}

}  // namespace grassland::vulkan
