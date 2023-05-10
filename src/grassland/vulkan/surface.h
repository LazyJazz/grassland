#pragma once
#include "instance.h"

namespace grassland::vulkan {
class Surface {
 public:
  Surface(const Instance &instance, GLFWwindow *window);
  ~Surface();

 private:
  GRASSLAND_VULKAN_HANDLE(VkSurfaceKHR, surface_);
  const Instance &instance_;
};
}  // namespace grassland::vulkan
