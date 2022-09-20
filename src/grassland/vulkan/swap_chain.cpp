#include <grassland/logging/logging.h>
#include <grassland/vulkan/swap_chain.h>

#include <algorithm>

namespace grassland::vulkan {

namespace {

VkSurfaceFormatKHR ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  for (const auto &availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB &&
        availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return availableFormat;
    }
  }

  return availableFormats[0];
}

VkPresentModeKHR ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &availablePresentModes) {
  for (const auto &availablePresentMode : availablePresentModes) {
    if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
      return availablePresentMode;
    }
  }
  // https://vulkan-tutorial.com/Drawing_a_triangle/Presentation/Swap_chain
  // Only the VK_PRESENT_MODE_FIFO_KHR mode is guaranteed to be available
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities,
                            GLFWwindow *window) {
  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  } else {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    VkExtent2D actualExtent = {static_cast<uint32_t>(width),
                               static_cast<uint32_t>(height)};

    actualExtent.width =
        std::clamp(actualExtent.width, capabilities.minImageExtent.width,
                   capabilities.maxImageExtent.width);
    actualExtent.height =
        std::clamp(actualExtent.height, capabilities.minImageExtent.height,
                   capabilities.maxImageExtent.height);

    return actualExtent;
  }
}
}  // namespace

SwapChain::SwapChain(GLFWwindow *window, Device *device) : handle_() {
  window_ = window;
  device_ = device;
  SwapChainSupportDetails swapChainSupport =
      device_->GetPhysicalDevice()->GetSwapChainSupportDetails(
          device_->GetSurface());

  VkSurfaceFormatKHR surfaceFormat =
      ChooseSwapSurfaceFormat(swapChainSupport.formats);
  VkPresentModeKHR presentMode =
      ChooseSwapPresentMode(swapChainSupport.presentModes);
  VkExtent2D extent = ChooseSwapExtent(swapChainSupport.capabilities, window_);
  ;
  uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      imageCount > swapChainSupport.capabilities.maxImageCount) {
    imageCount = swapChainSupport.capabilities.maxImageCount;
  }
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = device_->GetSurface()->GetHandle();
  createInfo.minImageCount = imageCount;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  uint32_t queueFamilyIndices[] = {
      device_->GetPhysicalDevice()->GraphicsFamilyIndex(),
      device_->GetPhysicalDevice()->PresentFamilyIndex(device_->GetSurface())};

  if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;      // Optional
    createInfo.pQueueFamilyIndices = nullptr;  // Optional
  }

  createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = presentMode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = VK_NULL_HANDLE;
  if (vkCreateSwapchainKHR(device->GetHandle(), &createInfo, nullptr,
                           &handle_) != VK_SUCCESS) {
    LAND_ERROR("Vulkan failed to create swap chain!");
  }

  swap_chain_extent_ = extent;
  swap_chain_image_format_ = surfaceFormat.format;
  image_count_ = imageCount;
  CreateImages();
  CreateImageViews();
}

SwapChain::~SwapChain() {
  image_views_.clear();
  vkDestroySwapchainKHR(device_->GetHandle(), GetHandle(), nullptr);
}

VkFormat SwapChain::GetFormat() const {
  return swap_chain_image_format_;
}

VkExtent2D SwapChain::GetExtent() const {
  return swap_chain_extent_;
}

uint32_t SwapChain::GetImageCount() const {
  return image_count_;
}

VkImage SwapChain::GetImage(uint32_t image_index) const {
  return images_[image_index];
}

void SwapChain::CreateImages() {
  std::vector<VkImage> swapChainImages;
  vkGetSwapchainImagesKHR(device_->GetHandle(), GetHandle(), &image_count_,
                          nullptr);
  swapChainImages.resize(image_count_);
  vkGetSwapchainImagesKHR(device_->GetHandle(), GetHandle(), &image_count_,
                          swapChainImages.data());
  for (int i = 0; i < image_count_; i++) {
    images_.emplace_back(swapChainImages[i]);
  }
}

void SwapChain::CreateImageViews() {
  image_views_.resize(images_.size());
  for (size_t i = 0; i < images_.size(); i++) {
    image_views_[i] = std::make_unique<ImageView>(device_, images_[i],
                                                  swap_chain_image_format_);
  }
}

ImageView *SwapChain::GetImageView(uint32_t image_index) const {
  return image_views_[image_index].get();
}

}  // namespace grassland::vulkan
