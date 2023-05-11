#include "swapchain.h"

#include "enumerate.h"

namespace grassland::vulkan {

Swapchain::Swapchain(const class Device &device, VkPresentModeKHR present_mode)
    : device_(device) {
  VkSurfaceCapabilitiesKHR capabilities{};
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device_.PhysicalDevice().Handle(),
                                            device_.Surface()->Handle(),
                                            &capabilities);
  auto formats = GetEnumerateVector(device_.PhysicalDevice().Handle(),
                                    device_.Surface()->Handle(),
                                    vkGetPhysicalDeviceSurfaceFormatsKHR);
  auto present_modes = GetEnumerateVector(
      device_.PhysicalDevice().Handle(), device_.Surface()->Handle(),
      vkGetPhysicalDeviceSurfacePresentModesKHR);

  const auto surface = device_.Surface();
  const auto window = surface->Window();

  const auto surface_format = ChooseSwapSurfaceFormat(formats);
  const auto actual_present_mode =
      ChooseSwapPresentMode(present_modes, present_mode);
  const auto extent = ChooseSwapExtent(window, capabilities);
  const auto image_count = ChooseImageCount(capabilities);

  VkSwapchainCreateInfoKHR createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  createInfo.surface = surface->Handle();
  createInfo.minImageCount = image_count;
  createInfo.imageFormat = surface_format.format;
  createInfo.imageColorSpace = surface_format.colorSpace;
  createInfo.imageExtent = extent;
  createInfo.imageArrayLayers = 1;
  createInfo.imageUsage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  createInfo.preTransform = capabilities.currentTransform;
  createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  createInfo.presentMode = actual_present_mode;
  createInfo.clipped = VK_TRUE;
  createInfo.oldSwapchain = nullptr;

  if (device.GraphicsFamilyIndex() != device.PresentFamilyIndex()) {
    uint32_t queueFamilyIndices[] = {device.GraphicsFamilyIndex(),
                                     device.PresentFamilyIndex()};

    createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    createInfo.queueFamilyIndexCount = 2;
    createInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;      // Optional
    createInfo.pQueueFamilyIndices = nullptr;  // Optional
  }

  GRASSLAND_VULKAN_CHECK(
      vkCreateSwapchainKHR(device.Handle(), &createInfo, nullptr, &swapchain_));
}

Swapchain::~Swapchain() {
  if (swapchain_) {
    vkDestroySwapchainKHR(device_.Handle(), swapchain_, nullptr);
  }
}

VkSurfaceFormatKHR Swapchain::ChooseSwapSurfaceFormat(
    const std::vector<VkSurfaceFormatKHR> &formats) {
  if (formats.size() == 1 && formats[0].format == VK_FORMAT_UNDEFINED) {
    return {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
  }

  for (const auto &format : formats) {
    if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
        format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
      return format;
    }
  }

  LAND_ERROR("No suitable surface format (VK_FORMAT_B8G8R8A8_UNORM)")
}

VkPresentModeKHR Swapchain::ChooseSwapPresentMode(
    const std::vector<VkPresentModeKHR> &present_modes,
    VkPresentModeKHR present_mode) {
  // VK_PRESENT_MODE_IMMEDIATE_KHR specifies that the presentation engine does
  // not wait for a vertical blanking period to update the current image,
  // meaning this mode may result in visible tearing. No internal queuing of
  // presentation requests is needed, as the requests are applied immediately.

  // VK_PRESENT_MODE_MAILBOX_KHR specifies that the presentation engine waits
  // for the next vertical blanking period to update the current image. Tearing
  // cannot be observed. An internal single-entry queue is used to hold pending
  // presentation requests. If the queue is full when a new presentation request
  // is received, the new request replaces the existing entry, and any images
  // associated with the prior entry become available for re-use by the
  // application. One request is removed from the queue and processed during
  // each vertical blanking period in which the queue is non-empty.

  // VK_PRESENT_MODE_FIFO_KHR specifies that the presentation engine waits for
  // the next vertical blanking period to update the current image. Tearing
  // cannot be observed. An internal queue is used to hold pending presentation
  // requests. New requests are appended to the end of the queue, and one
  // request is removed from the beginning of the queue and processed during
  // each vertical blanking period in which the queue is non-empty. This is the
  // only value of presentMode that is required to be supported.

  // VK_PRESENT_MODE_FIFO_RELAXED_KHR specifies that the presentation engine
  // generally waits for the next vertical blanking period to update the current
  // image. If a vertical blanking period has already passed since the last
  // update of the current image then the presentation engine does not wait for
  // another vertical blanking period for the update, meaning this mode may
  // result in visible tearing in this case. This mode is useful for reducing
  // visual stutter with an application that will mostly present a new image
  // before the next vertical blanking period, but may occasionally be late, and
  // present a new image just after the next vertical blanking period. An
  // internal queue is used to hold pending presentation requests. New requests
  // are appended to the end of the queue, and one request is removed from the
  // beginning of the queue and processed during or after each vertical blanking
  // period in which the queue is non-empty.

  switch (present_mode) {
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
    case VK_PRESENT_MODE_MAILBOX_KHR:
    case VK_PRESENT_MODE_FIFO_KHR:
    case VK_PRESENT_MODE_FIFO_RELAXED_KHR:

      if (std::find(present_modes.begin(), present_modes.end(), present_mode) !=
          present_modes.end()) {
        return present_mode;
      }

      break;

    default:
      LAND_ERROR("unknown present mode")
  }

  // Fallback
  return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D Swapchain::ChooseSwapExtent(
    GLFWwindow *window,
    const VkSurfaceCapabilitiesKHR &capabilities) {
  // Vulkan tells us to match the resolution of the window by setting the width
  // and height in the currentExtent member. However, some window managers do
  // allow us to differ here and this is indicated by setting the width and
  // height in currentExtent to a special value: the maximum value of uint32_t.
  // In that case we'll pick the resolution that best matches the window within
  // the minImageExtent and maxImageExtent bounds.

  if (capabilities.currentExtent.width !=
      std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  VkExtent2D actualExtent{static_cast<uint32_t>(width),
                          static_cast<uint32_t>(height)};

  actualExtent.width =
      std::max(capabilities.minImageExtent.width,
               std::min(capabilities.maxImageExtent.width, actualExtent.width));
  actualExtent.height = std::max(
      capabilities.minImageExtent.height,
      std::min(capabilities.maxImageExtent.height, actualExtent.height));

  return actualExtent;
}

uint32_t Swapchain::ChooseImageCount(
    const VkSurfaceCapabilitiesKHR &capabilities) {
  // The implementation specifies the minimum amount of images to function
  // properly and we'll try to have one more than that to properly implement
  // triple buffering. (tanguyf: or not, we can just rely on
  // VK_PRESENT_MODE_MAILBOX_KHR with two buffers)
  uint32_t imageCount = std::max(2u, capabilities.minImageCount);  // +1;

  if (capabilities.maxImageCount > 0 &&
      imageCount > capabilities.maxImageCount) {
    imageCount = capabilities.maxImageCount;
  }

  return imageCount;
}

}  // namespace grassland::vulkan
