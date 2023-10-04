#include "grassland/vulkan/core/swap_chain.h"

#include "grassland/vulkan/core/single_time_commands.h"

namespace grassland::vulkan {
namespace {

SwapChainSupportDetails QuerySwapChainSupport(VkPhysicalDevice device,
                                              VkSurfaceKHR surface) {
  SwapChainSupportDetails details;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface,
                                            &details.capabilities);
  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  if (formatCount != 0) {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount,
                                         details.formats.data());
  }
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount,
                                            nullptr);

  if (presentModeCount != 0) {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device, surface, &presentModeCount, details.presentModes.data());
  }
  return details;
}

VkSurfaceFormatKHR ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &availableFormats) {
  for (const auto &availableFormat : availableFormats) {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM &&
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

SwapChain::SwapChain(Device *device, Surface *surface)
    : device_(device), surface_(surface) {
  CreateSwapChain();
  CreateImageViews();
}

void SwapChain::CreateSwapChain() {
  SwapChainSupportDetails swapChainSupport = QuerySwapChainSupport(
      device_->PhysicalDevice().Handle(), surface_->Handle());

  // List all available surface formats, print both VkFormat and VkColorSpace
  spdlog::info("Available surface formats:");
  for (const auto &format : swapChainSupport.formats) {
    spdlog::info("  Format: {}, Color space: {}", VkFormatToName(format.format),
                 VkColorSpaceToName(format.colorSpace));
  }

  VkSurfaceFormatKHR surfaceFormat =
      ChooseSwapSurfaceFormat(swapChainSupport.formats);
  VkPresentModeKHR presentMode =
      ChooseSwapPresentMode(swapChainSupport.presentModes);
  VkExtent2D extent =
      ChooseSwapExtent(swapChainSupport.capabilities, surface_->Window());

  spdlog::info("Swap chain extent: {}x{}", extent.width, extent.height);
  spdlog::info("Swap chain format: {}", VkFormatToName(surfaceFormat.format));
  spdlog::info("Swap chain color space: {}",
               VkColorSpaceToName(surfaceFormat.colorSpace));
  spdlog::info("Swap chain present mode: {}", VkPresentModeToName(presentMode));

  // Print selected surface format and present mode

  uint32_t image_count = swapChainSupport.capabilities.minImageCount + 1;
  if (swapChainSupport.capabilities.maxImageCount > 0 &&
      image_count > swapChainSupport.capabilities.maxImageCount) {
    image_count = swapChainSupport.capabilities.maxImageCount;
  }
  VkSwapchainCreateInfoKHR createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface_->Handle();
  createInfo.minImageCount = image_count;
  createInfo.imageFormat = surfaceFormat.format;
  createInfo.imageColorSpace = surfaceFormat.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  uint32_t queueFamilyIndices[] = {
      device_->PhysicalDevice().GraphicsFamilyIndex(),
      device_->PhysicalDevice().PresentFamilyIndex(surface_)};

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
  if (vkCreateSwapchainKHR(device_->Handle(), &createInfo, nullptr,
                           &swap_chain_) != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create swap chain!");
  }

  extent_ = extent;
  format_ = surfaceFormat.format;
}

void SwapChain::CreateImageViews() {
  // Get images and imageviews from swapchain
  vkGetSwapchainImagesKHR(device_->Handle(), swap_chain_, &image_count_,
                          nullptr);
  images_.resize(image_count_);
  vkGetSwapchainImagesKHR(device_->Handle(), swap_chain_, &image_count_,
                          images_.data());
  image_views_.resize(image_count_);

  for (size_t i = 0; i < image_count_; i++) {
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = images_[i];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format_;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(device_->Handle(), &createInfo, nullptr,
                          &image_views_[i]) != VK_SUCCESS) {
      LAND_ERROR("[Vulkan] failed to create image views!");
    }
  }
}

SwapChain::~SwapChain() {
  for (auto &image_view : image_views_) {
    vkDestroyImageView(device_->Handle(), image_view, nullptr);
  }
  vkDestroySwapchainKHR(device_->Handle(), swap_chain_, nullptr);
}

VkResult SwapChain::AcquireNextImage(uint32_t *image_index,
                                     VkSemaphore semaphore,
                                     VkFence fence) {
  return vkAcquireNextImageKHR(device_->Handle(), swap_chain_,
                               std::numeric_limits<uint64_t>::max(), semaphore,
                               fence, image_index);
}

}  // namespace grassland::vulkan
