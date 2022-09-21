#pragma once
#include <grassland/vulkan/device.h>
#include <grassland/vulkan/image.h>
#include <grassland/vulkan/image_view.h>

namespace grassland::vulkan {
class SwapChain {
 public:
  SwapChain(GLFWwindow *window, Device *device);
  ~SwapChain();
  [[nodiscard]] VkFormat GetFormat() const;
  [[nodiscard]] VkExtent2D GetExtent() const;
  [[nodiscard]] uint32_t GetImageCount() const;
  [[nodiscard]] VkImage GetImage(uint32_t image_index) const;
  [[nodiscard]] ImageView *GetImageView(uint32_t image_index) const;

 private:
  void CreateImages();
  void CreateImageViews();

  GRASSLAND_VULKAN_HANDLE(VkSwapchainKHR)
  GRASSLAND_VULKAN_DEVICE_PTR
  GLFWwindow *window_{nullptr};
  VkFormat swap_chain_image_format_{};
  VkExtent2D swap_chain_extent_{};
  std::vector<VkImage> images_;
  std::vector<std::unique_ptr<ImageView>> image_views_;
  uint32_t image_count_{0};
};
}  // namespace grassland::vulkan
