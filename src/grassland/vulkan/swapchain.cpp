#include "swapchain.h"

#include "enumerate.h"

namespace grassland::vulkan {

Swapchain::Swapchain(const class Device &device) : device_(device) {
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
}

}  // namespace grassland::vulkan
