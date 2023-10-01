#pragma once
#include "grassland/vulkan/core/device.h"
#include "grassland/vulkan/core/surface.h"

namespace grassland::vulkan {
class SwapChain {
 public:
  SwapChain(Device *device, Surface *surface);
  ~SwapChain();

  VkSwapchainKHR Handle() {
    return swap_chain_;
  }
  VkFormat Format() {
    return format_;
  }
  VkExtent2D Extent() {
    return extent_;
  }
  std::vector<VkImage> Images() {
    return images_;
  }
  std::vector<VkImageView> ImageViews() {
    return image_views_;
  }

 private:
  void CreateSwapChain();
  void CreateImageViews();

  Device *device_;
  Surface *surface_;

  VkSwapchainKHR swap_chain_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;
};
}  // namespace grassland::vulkan
