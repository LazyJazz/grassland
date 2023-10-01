#pragma once
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "grassland/vulkan/core/instance.h"
#include "grassland/vulkan/utils/vulkan_utils.h"

namespace grassland::vulkan {
class Surface {
 public:
  Surface(Instance *instance, GLFWwindow *window);
  ~Surface();

  [[nodiscard]] VkSurfaceKHR Handle() const;

 private:
  Instance *instance_{};
  VkSurfaceKHR surface_{};
};
}  // namespace grassland::vulkan
