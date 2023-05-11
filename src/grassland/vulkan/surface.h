#pragma once
#include "instance.h"

namespace grassland::vulkan {
class Surface {
 public:
  GRASSLAND_CANNOT_COPY(Surface);
  Surface(const Instance &instance, GLFWwindow *window);
  ~Surface();

  [[nodiscard]] GLFWwindow *Window() const {
    return window_;
  }

 private:
  GRASSLAND_VULKAN_HANDLE(VkSurfaceKHR, surface_);
  const Instance &instance_;
  GLFWwindow *window_{};
};
}  // namespace grassland::vulkan
