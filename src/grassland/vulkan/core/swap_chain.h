#pragma once
#include "grassland/vulkan/core/device.h"
#include "grassland/vulkan/core/surface.h"

namespace grassland::vulkan {

struct SwapChainSupportDetails {
  VkSurfaceCapabilitiesKHR capabilities;
  std::vector<VkSurfaceFormatKHR> formats;
  std::vector<VkPresentModeKHR> presentModes;
};

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

  // Acquire an image from the swap chain
  VkResult AcquireNextImage(uint32_t *image_index,
                            VkSemaphore semaphore,
                            VkFence fence);

 private:
  void CreateSwapChain();
  void CreateImageViews();

  Device *device_;
  Surface *surface_;

  VkSwapchainKHR swap_chain_{};
  VkFormat format_{};
  VkExtent2D extent_{};
  uint32_t image_count_{};
  std::vector<VkImage> images_;
  std::vector<VkImageView> image_views_;
};
}  // namespace grassland::vulkan
