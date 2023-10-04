#include "grassland/vulkan/core/surface.h"

namespace grassland::vulkan {
// Complete the implementation of the Surface class here.
Surface::Surface(Instance *instance, GLFWwindow *window)
    : instance_(instance), window_(window) {
  VkResult result =
      glfwCreateWindowSurface(instance->Handle(), window, nullptr, &surface_);
  if (result != VK_SUCCESS) {
    LAND_ERROR("[Vulkan] failed to create window surface!");
  }
}

Surface::~Surface() {
  vkDestroySurfaceKHR(instance_->Handle(), surface_, nullptr);
}

VkSurfaceKHR Surface::Handle() const {
  return surface_;
}

GLFWwindow *Surface::Window() const {
  return window_;
}

}  // namespace grassland::vulkan
