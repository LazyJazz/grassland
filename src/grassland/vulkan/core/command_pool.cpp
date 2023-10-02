#include "command_pool.h"

namespace grassland::vulkan {
CommandPool::CommandPool(class Device *device) : device_(device) {
  VkCommandPoolCreateInfo command_pool_create_info = {};
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.queueFamilyIndex =
      device_->PhysicalDevice().GraphicsFamilyIndex();
  command_pool_create_info.flags =
      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  if (vkCreateCommandPool(device_->Handle(), &command_pool_create_info, nullptr,
                          &command_pool_) != VK_SUCCESS) {
    throw std::runtime_error("failed to create command pool!");
  }
}

CommandPool::~CommandPool() {
  vkDestroyCommandPool(device_->Handle(), command_pool_, nullptr);
}

VkCommandPool CommandPool::Handle() const {
  return command_pool_;
}

class Device *CommandPool::Device() const {
  return device_;
}
}  // namespace grassland::vulkan
