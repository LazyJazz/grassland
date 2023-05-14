#pragma once
#include "device.h"
#include "image_view.h"

namespace grassland::vulkan {
class Swapchain {
 public:
  GRASSLAND_CANNOT_COPY(Swapchain)
  explicit Swapchain(
      const class Device &device,
      VkPresentModeKHR present_mode = VK_PRESENT_MODE_IMMEDIATE_KHR);
  ~Swapchain();

  [[nodiscard]] VkFormat Format() const {
    return format_;
  }
  [[nodiscard]] VkExtent2D Extent() const {
    return extent_;
  }
  [[nodiscard]] VkPresentModeKHR PresentMode() const {
    return present_mode_;
  }
  [[nodiscard]] VkImage Image(size_t index) const {
    return images_[index];
  }
  [[nodiscard]] const class ImageView &ImageView(size_t index) const {
    return *image_views_[index];
  }

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

  VkPresentModeKHR present_mode_;
  VkFormat format_;
  VkExtent2D extent_;
  std::vector<VkImage> images_;
  std::vector<std::unique_ptr<class ImageView>> image_views_;
};
}  // namespace grassland::vulkan
