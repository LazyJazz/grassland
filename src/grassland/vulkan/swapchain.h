#pragma once
#include "device.h"

namespace grassland::vulkan {
class Swapchain {
 public:
  GRASSLAND_CANNOT_COPY(Swapchain)
  explicit Swapchain(
      const class Device &device,
      VkPresentModeKHR present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR);
  ~Swapchain();

 private:
  static VkSurfaceFormatKHR ChooseSwapSurfaceFormat(
      const std::vector<VkSurfaceFormatKHR> &formats);
  static VkPresentModeKHR ChooseSwapPresentMode(
      const std::vector<VkPresentModeKHR> &present_modes,
      VkPresentModeKHR present_mode);
  static VkExtent2D ChooseSwapExtent(
      GLFWwindow *window,
      const VkSurfaceCapabilitiesKHR &capabilities);
  static uint32_t ChooseImageCount(
      const VkSurfaceCapabilitiesKHR &capabilities);

  GRASSLAND_VULKAN_HANDLE(VkSwapchainKHR, swapchain_)
  GRASSLAND_VULKAN_DEVICE
};
}  // namespace grassland::vulkan
