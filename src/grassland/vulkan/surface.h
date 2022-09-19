#pragma once
#include <GLFW/glfw3.h>
#include <grassland/vulkan/instance.h>
#include <grassland/vulkan/util.h>

namespace grassland::vulkan {
class Surface {
 public:
  explicit Surface(Instance *instance, GLFWwindow *window);
  ~Surface();

 private:
  VK_HANDLE(VkSurfaceKHR)
  Instance *instance_;
};
}  // namespace grassland::vulkan
