#include "surface.h"

namespace grassland::vulkan {

Surface::Surface(const Instance &instance, GLFWwindow *window)
    : instance_(instance) {
  GRASSLAND_VULKAN_CHECK(
      glfwCreateWindowSurface(instance.Handle(), window, nullptr, &surface_));
}

Surface::~Surface() {
  vkDestroySurfaceKHR(instance_.Handle(), surface_, nullptr);
}

}  // namespace grassland::vulkan
