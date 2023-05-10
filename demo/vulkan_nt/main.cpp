#include "grassland/grassland.h"
#include "iostream"

using namespace grassland::vulkan;

int main() {
  glfwInit();
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  auto window = glfwCreateWindow(800, 600, "Grassland", nullptr, nullptr);
  Instance instance{InstanceSettings{true, true}};
  Surface surface(instance, window);
  auto handle = instance.Handle();
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
  }
}
