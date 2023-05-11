#include "buffer.h"

namespace grassland::vulkan {

Buffer::Buffer(const class Device &device,
               size_t size,
               VkBufferUsageFlags usage)
    : device_(device) {
  VkBufferCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  create_info.size = size;
  create_info.usage = usage;
  create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  GRASSLAND_VULKAN_CHECK(
      vkCreateBuffer(device_.Handle(), &create_info, nullptr, &buffer_));
}

Buffer::~Buffer() {
  if (buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_.Handle(), buffer_, nullptr);
  }
}

DeviceMemory Buffer::AllocateMemory(VkMemoryPropertyFlags property_flags) {
  return AllocateMemory(0, property_flags);
}

DeviceMemory Buffer::AllocateMemory(VkMemoryAllocateFlags allocate_flags,
                                    VkMemoryPropertyFlags property_flags) {
  const auto requirements = GetMemoryRequirements();
  DeviceMemory memory(device_, requirements.size, requirements.memoryTypeBits,
                      allocate_flags, property_flags);
  GRASSLAND_VULKAN_CHECK(
      vkBindBufferMemory(device_.Handle(), buffer_, memory.Handle(), 0));
  return memory;
}

VkMemoryRequirements Buffer::GetMemoryRequirements() const {
  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device_.Handle(), buffer_, &requirements);
  return requirements;
}

VkDeviceAddress Buffer::GetDeviceAddress() const {
  VkBufferDeviceAddressInfo address_info{};
  address_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
  address_info.buffer = buffer_;
  return vkGetBufferDeviceAddress(device_.Handle(), &address_info);
}

}  // namespace grassland::vulkan
