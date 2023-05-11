#include "command_pool.h"

namespace grassland::vulkan {
CommandPool::CommandPool(const class Device &device)
    : CommandPool(device, device.GraphicsFamilyIndex()) {
}

CommandPool::CommandPool(const class Device &device, int queue_family_index)
    : device_(device) {
  VkCommandPoolCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  create_info.queueFamilyIndex = queue_family_index;
  create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  GRASSLAND_VULKAN_CHECK(vkCreateCommandPool(device_.Handle(), &create_info,
                                             nullptr, &command_pool_));
}

CommandPool::~CommandPool() {
  vkDestroyCommandPool(device_.Handle(), command_pool_, nullptr);
}

}  // namespace grassland::vulkan
