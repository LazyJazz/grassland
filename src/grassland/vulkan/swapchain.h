#pragma once
#include "device.h"

namespace grassland::vulkan {
class Swapchain {
 public:
  GRASSLAND_CANNOT_COPY(Swapchain)
  explicit Swapchain(const class Device &device);

 private:
  GRASSLAND_VULKAN_HANDLE(VkSwapchainKHR, swapchain_)
  GRASSLAND_VULKAN_DEVICE
};
}  // namespace grassland::vulkan
