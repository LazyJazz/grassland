#pragma once
#include <grassland/vulkan/device.h>

namespace grassland::vulkan {
class SwapChain {
 public:
  SwapChain(GLFWwindow *window, Device *device);
  ~SwapChain();

 private:
  VK_HANDLE(VkSwapchainKHR)
  Device *device_{nullptr};
  GLFWwindow *window_{nullptr};
};
}  // namespace grassland::vulkan
